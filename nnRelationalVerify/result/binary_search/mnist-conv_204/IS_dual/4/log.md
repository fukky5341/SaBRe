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
execution time: IAR + LP analysis = 12.99 + 35.16 = 48.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.6198275, upper bound: 2.6198270


# Binary Search by BASE starts (time budget: 3551.85 seconds, max iter: 100)

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
Binary search time: 208.29 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual) starts
Time budget: 3343.56 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3247599
time: 5.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3480098
time: 6.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.09
Output dim: 8, lower bound: -2.3480088, upper bound: 2.3247599
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.09
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

Time for backsubstitution: 13.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3247565
time: 6.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3247559
time: 6.37 seconds

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
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3480091
time: 8.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3480105
time: 6.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 8, lower bound: -2.3247561, upper bound: 2.3247565
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3247559
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.42
Output dim: 8, lower bound: -2.3247562, upper bound: 2.3480091
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.42
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

Time for backsubstitution: 12.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221806, upper bound: 2.3247572
time: 5.23 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247530, upper bound: 2.3247569
time: 8.72 seconds

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

Time for backsubstitution: 13.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221806, upper bound: 2.3247562
time: 6.07 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247529, upper bound: 2.3247582
time: 6.29 seconds

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

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3454424
time: 5.59 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3480050
time: 5.31 seconds

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

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221806, upper bound: 2.3480055
time: 5.51 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247529, upper bound: 2.3480059
time: 6.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.12 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3221806, upper bound: 2.3247572
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3247530, upper bound: 2.3247569
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3221806, upper bound: 2.3247562
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3247529, upper bound: 2.3247582
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3454424
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3480050
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3221806, upper bound: 2.3480055
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.12
Output dim: 8, lower bound: -2.3247529, upper bound: 2.3480059

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -6.1852236, -1.7829347, -6.1710052, -1.8091354, -4.3760881, 4.3880706
1: -15.2631798, -10.7432575, -15.2505445, -10.7507181, -4.5124617, 4.5072870
2: -9.0590172, -4.6691799, -9.0504265, -4.6823287, -4.2772779, 4.2824059
3: -7.5695829, -3.6011109, -7.5379372, -3.6306880, -3.9388950, 3.9368262
4: -12.2054796, -7.3996406, -12.1911879, -7.4107089, -4.7947707, 4.7915473
5: -5.9826055, -2.2345085, -5.9681535, -2.2458134, -3.7367921, 3.7336450
6: -13.7674370, -9.3107300, -13.7295284, -9.3505440, -4.1065369, 4.1069636
7: -10.2198896, -5.8791976, -10.1899185, -5.9047527, -4.3151369, 4.3107209
8: 7.8692045, 11.0702295, 7.8826990, 11.0574713, -3.1882668, 3.1867666
9: -7.1358552, -3.2471225, -7.1110320, -3.2506745, -3.7400122, 3.7183204

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216867, upper bound: 2.3113501
time: 5.33 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221634, upper bound: 2.3247358
time: 5.19 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.1879406, -1.7817001, -4.4062433, 4.4062428
1: -15.2639427, -10.7425175, -15.2639418, -10.7425203, -4.5214224, 4.5214243
2: -9.0594845, -4.6674333, -9.0594826, -4.6674356, -4.2938032, 4.2939396
3: -7.5704598, -3.5940862, -7.5704584, -3.5940981, -3.9763618, 3.9763722
4: -12.2075682, -7.3990154, -12.2075644, -7.3990145, -4.8085537, 4.8085489
5: -5.9838700, -2.2321615, -5.9838691, -2.2321656, -3.7517045, 3.7517076
6: -13.7705059, -9.3012829, -13.7705040, -9.3012981, -4.1249676, 4.1584945
7: -10.2260075, -5.8782048, -10.2259922, -5.8782072, -4.3478003, 4.3477874
8: 7.8683748, 11.0728474, 7.8683772, 11.0728416, -3.2044668, 3.2041516
9: -7.1385541, -3.2466414, -7.1385517, -3.2466414, -3.7496281, 3.7495599

Time for backsubstitution: 13.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247569, upper bound: 2.3221801
time: 7.02 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247569, upper bound: 2.3247572
time: 6.50 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -6.1852236, -1.7829347, -6.1962962, -1.7841535, -4.4010701, 4.4133615
1: -15.2631798, -10.7432575, -15.2972927, -10.7122021, -4.5509777, 4.5540352
2: -9.0590172, -4.6691799, -9.1495380, -4.5761046, -4.3840837, 4.3848314
3: -7.5695829, -3.6011109, -7.5837145, -3.5838249, -3.9857581, 3.9826035
4: -12.2054796, -7.3996406, -12.2653923, -7.3659816, -4.8394980, 4.8657517
5: -5.9826055, -2.2345085, -6.0167847, -2.2000177, -3.7825878, 3.7822762
6: -13.7674370, -9.3107300, -13.7736197, -9.2978554, -4.1665297, 4.1482964
7: -10.2198896, -5.8791976, -10.2340288, -5.8906298, -4.3292599, 4.3548312
8: 7.8692045, 11.0702295, 7.8268032, 11.1009731, -3.2317686, 3.2434263
9: -7.1358552, -3.2471225, -7.1479387, -3.2283797, -3.8631158, 3.7581968

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3449758, upper bound: 2.3113495
time: 5.16 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454207, upper bound: 2.3247353
time: 5.93 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -6.1879435, -1.7816978, -6.2133961, -1.7566996, -4.4312439, 4.4316983
1: -15.2639427, -10.7425175, -15.3121748, -10.7039547, -4.5599880, 4.5696573
2: -9.0594845, -4.6674333, -9.1586056, -4.5603304, -4.4004035, 4.3963809
3: -7.5704598, -3.5940862, -7.6162853, -3.5472577, -4.0232019, 4.0221992
4: -12.2075682, -7.3990154, -12.2817230, -7.3541961, -4.8533721, 4.8827076
5: -5.9838700, -2.2321615, -6.0326071, -2.1862943, -3.7975757, 3.8004456
6: -13.7705059, -9.3012829, -13.8142920, -9.2481222, -4.1873264, 4.1854172
7: -10.2260075, -5.8782048, -10.2709503, -5.8640852, -4.3619223, 4.3927455
8: 7.8683748, 11.0728474, 7.8114200, 11.1164789, -3.2481041, 3.2614274
9: -7.1385541, -3.2466414, -7.1753473, -3.2234244, -3.8728042, 3.7894020

Time for backsubstitution: 16.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480047, upper bound: 2.3221794
time: 6.91 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3480047, upper bound: 2.3247562
time: 5.98 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1962962, -1.7841535, -6.1852236, -1.7829347, -4.4133615, 4.4010701
1: -15.2972927, -10.7122021, -15.2631798, -10.7432575, -4.5540352, 4.5509777
2: -9.1495380, -4.5761046, -9.0590172, -4.6691799, -4.3848314, 4.3840828
3: -7.5837145, -3.5838249, -7.5695829, -3.6011109, -3.9826035, 3.9857581
4: -12.2653923, -7.3659816, -12.2054796, -7.3996406, -4.8657517, 4.8394980
5: -6.0167847, -2.2000177, -5.9826055, -2.2345085, -3.7822762, 3.7825878
6: -13.7736197, -9.2978554, -13.7674370, -9.3107300, -4.1482964, 4.1665297
7: -10.2340288, -5.8906298, -10.2198896, -5.8791976, -4.3548312, 4.3292599
8: 7.8268032, 11.1009731, 7.8692045, 11.0702295, -3.2434263, 3.2317686
9: -7.1479387, -3.2283797, -7.1358552, -3.2471225, -3.7581968, 3.8631153

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3113495, upper bound: 2.3449760
time: 5.19 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247352, upper bound: 2.3454213
time: 4.90 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2133961, -1.7566996, -6.1879435, -1.7816978, -4.4316983, 4.4312439
1: -15.3121748, -10.7039547, -15.2639427, -10.7425175, -4.5696573, 4.5599880
2: -9.1586056, -4.5603304, -9.0594845, -4.6674333, -4.3963814, 4.4004035
3: -7.6162853, -3.5472577, -7.5704598, -3.5940862, -4.0221992, 4.0232019
4: -12.2817230, -7.3541961, -12.2075682, -7.3990154, -4.8827076, 4.8533721
5: -6.0326071, -2.1862943, -5.9838700, -2.2321615, -3.8004456, 3.7975757
6: -13.8142920, -9.2481222, -13.7705059, -9.3012829, -4.1854172, 4.1873264
7: -10.2709503, -5.8640852, -10.2260075, -5.8782048, -4.3927455, 4.3619223
8: 7.8114200, 11.1164789, 7.8683748, 11.0728474, -3.2614274, 3.2481041
9: -7.1753473, -3.2234244, -7.1385541, -3.2466414, -3.7894025, 3.8728027

Time for backsubstitution: 13.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221793, upper bound: 2.3480053
time: 5.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221793, upper bound: 2.3480052
time: 5.45 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -6.2106447, -1.7579322, -6.1962962, -1.7841535, -4.4264913, 4.4383640
1: -15.3113861, -10.7046852, -15.2972927, -10.7122021, -4.5991840, 4.5926075
2: -9.1581306, -4.5620828, -9.1495380, -4.5761046, -4.4723892, 4.4779749
3: -7.6154127, -3.5542626, -7.5837145, -3.5838249, -4.0315876, 4.0294518
4: -12.2796526, -7.3548303, -12.2653923, -7.3659816, -4.9136710, 4.9105620
5: -6.0313563, -2.1886551, -6.0167847, -2.2000177, -3.8313386, 3.8281295
6: -13.8112803, -9.2575436, -13.7736197, -9.2978554, -4.1866283, 4.1878433
7: -10.2648516, -5.8650646, -10.2340288, -5.8906298, -4.3742218, 4.3689642
8: 7.8122592, 11.1138420, 7.8268032, 11.1009731, -3.2887139, 3.2870388
9: -7.1726923, -3.2238898, -7.1479387, -3.2283797, -3.9252796, 3.9036021

Time for backsubstitution: 13.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221758, upper bound: 2.3454371
time: 7.61 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221758, upper bound: 2.3480061
time: 6.09 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -6.2133999, -1.7566977, -6.2133961, -1.7566996, -4.4567003, 4.4566984
1: -15.3121786, -10.7039528, -15.3121748, -10.7039547, -4.6082239, 4.6082220
2: -9.1586056, -4.5603261, -9.1586056, -4.5603304, -4.4887204, 4.4880171
3: -7.6162882, -3.5472469, -7.6162853, -3.5472577, -4.0690308, 4.0690384
4: -12.2817307, -7.3541937, -12.2817230, -7.3541961, -4.9275346, 4.9275293
5: -6.0326066, -2.1862900, -6.0326071, -2.1862943, -3.8463123, 3.8463171
6: -13.8142948, -9.2481070, -13.8142920, -9.2481222, -4.2073736, 4.2391253
7: -10.2709599, -5.8640842, -10.2709503, -5.8640852, -4.4068747, 4.4068661
8: 7.8114176, 11.1164827, 7.8114200, 11.1164789, -3.3050613, 3.3050628
9: -7.1753492, -3.2234240, -7.1753473, -3.2234244, -3.9348135, 3.9398613

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3454387
time: 4.53 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3480050
time: 19.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 37.34 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3216867, upper bound: 2.3113501
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3221634, upper bound: 2.3247358
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3247569, upper bound: 2.3221801
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3247569, upper bound: 2.3247572
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3449758, upper bound: 2.3113495
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3454207, upper bound: 2.3247353
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3480047, upper bound: 2.3221794
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3480047, upper bound: 2.3247562
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3113495, upper bound: 2.3449760
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3247352, upper bound: 2.3454213
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3221793, upper bound: 2.3480053
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3221793, upper bound: 2.3480052
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3221758, upper bound: 2.3454371
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3221758, upper bound: 2.3480061
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3454387
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 37.34
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3480050

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -6.1736069, -1.7850957, -6.1108990, -1.8464925, -4.3271141, 4.3258033
1: -15.2520123, -10.7452097, -15.1916485, -10.7787066, -4.4733057, 4.4464388
2: -9.0580978, -4.6760082, -9.0386009, -4.7194743, -4.2327175, 4.2612543
3: -7.5674992, -3.6029730, -7.5195389, -3.6422558, -3.9252434, 3.9165659
4: -12.2015228, -7.4047403, -12.1512642, -7.4365139, -4.7650089, 4.7465239
5: -5.9800134, -2.2374291, -5.9469881, -2.2628121, -3.7172012, 3.7095590
6: -13.7589760, -9.3125343, -13.6862612, -9.3759060, -4.0728903, 4.0608454
7: -10.2175169, -5.8833332, -10.1658173, -5.9266062, -4.2909107, 4.2824841
8: 7.8747721, 11.0684881, 7.9118772, 11.0355921, -3.1608200, 3.1554818
9: -7.1318150, -3.2500587, -7.0880852, -3.2687817, -3.7178435, 3.6868730

Time for backsubstitution: 13.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216868, upper bound: 2.3087074
time: 5.35 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216867, upper bound: 2.3113501
time: 5.38 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -6.1852236, -1.7829347, -6.1709595, -1.8091455, -4.3760781, 4.3880248
1: -15.2631798, -10.7432575, -15.2504911, -10.7507257, -4.5124540, 4.5072336
2: -9.0590172, -4.6691799, -9.0504208, -4.6823602, -4.2696075, 4.2824006
3: -7.5695829, -3.6011109, -7.5379286, -3.6306965, -3.9388864, 3.9368176
4: -12.2054796, -7.3996406, -12.1911669, -7.4107237, -4.7947559, 4.7915263
5: -5.9826055, -2.2345085, -5.9681416, -2.2458189, -3.7367866, 3.7336330
6: -13.7674370, -9.3107300, -13.7294884, -9.3505507, -4.1065292, 4.1051292
7: -10.2198896, -5.8791976, -10.1899071, -5.9047737, -4.3151159, 4.3107095
8: 7.8692045, 11.0702295, 7.8827248, 11.0574627, -3.1882582, 3.1783133
9: -7.1358552, -3.2471225, -7.1110163, -3.2506840, -3.7399893, 3.7284141

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221635, upper bound: 2.3221587
time: 5.41 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221634, upper bound: 2.3247358
time: 5.32 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1710052, -1.8091354, -6.1879406, -1.7817001, -4.3893051, 4.3788052
1: -15.2505445, -10.7507181, -15.2639418, -10.7425203, -4.5080242, 4.5132236
2: -9.0504265, -4.6823287, -9.0594826, -4.6674356, -4.2839050, 4.2770410
3: -7.5379372, -3.6306880, -7.5704584, -3.5940981, -3.9438391, 3.9397705
4: -12.1911879, -7.4107089, -12.2075644, -7.3990145, -4.7921734, 4.7968554
5: -5.9681535, -2.2458134, -5.9838691, -2.2321656, -3.7359879, 3.7380557
6: -13.7295284, -9.3505440, -13.7705040, -9.3012981, -4.1161013, 4.1096430
7: -10.1899185, -5.9047527, -10.2259922, -5.8782072, -4.3117113, 4.3212395
8: 7.8826990, 11.0574713, 7.8683772, 11.0728416, -3.1887379, 3.1890941
9: -7.1110320, -3.2506745, -7.1385517, -3.2466414, -3.7180533, 3.7428970

Time for backsubstitution: 13.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3087072, upper bound: 2.3216813
time: 8.47 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221582, upper bound: 2.3221595
time: 4.88 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1879406, -1.7817001, -6.1879406, -1.7817001, -4.4062405, 4.4062405
1: -15.2639418, -10.7425203, -15.2639418, -10.7425203, -4.5214214, 4.5214214
2: -9.0594826, -4.6674356, -9.0594826, -4.6674356, -4.2939367, 4.2939363
3: -7.5704584, -3.5940981, -7.5704584, -3.5940981, -3.9763603, 3.9763603
4: -12.2075644, -7.3990145, -12.2075644, -7.3990145, -4.8085499, 4.8085499
5: -5.9838691, -2.2321656, -5.9838691, -2.2321656, -3.7517035, 3.7517035
6: -13.7705040, -9.3012981, -13.7705040, -9.3012981, -4.1249628, 4.1249633
7: -10.2259922, -5.8782072, -10.2259922, -5.8782072, -4.3477850, 4.3477850
8: 7.8683772, 11.0728416, 7.8683772, 11.0728416, -3.2044644, 3.2044644
9: -7.1385517, -3.2466414, -7.1385517, -3.2466414, -3.7496243, 3.7496243

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216819, upper bound: 2.3113496
time: 5.62 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221588, upper bound: 2.3247355
time: 5.28 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -6.1736069, -1.7850957, -6.1362791, -1.8214746, -4.3521323, 4.3511834
1: -15.2520123, -10.7452097, -15.2382622, -10.7401867, -4.5118256, 4.4930525
2: -9.0580978, -4.6760082, -9.1377182, -4.6134310, -4.3397894, 4.3636951
3: -7.5674992, -3.6029730, -7.5652986, -3.5954003, -3.9720988, 3.9623256
4: -12.2015228, -7.4047403, -12.2253914, -7.3918042, -4.8097186, 4.8206511
5: -5.9800134, -2.2374291, -5.9956317, -2.2169721, -3.7630413, 3.7582026
6: -13.7589760, -9.3125343, -13.7303085, -9.3233337, -4.1323509, 4.1020036
7: -10.2175169, -5.8833332, -10.2096767, -5.9124813, -4.3050356, 4.3263435
8: 7.8747721, 11.0684881, 7.8561664, 11.0790901, -3.2043180, 3.2123218
9: -7.1318150, -3.2500587, -7.1246834, -3.2466125, -3.8404264, 3.7266493

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3449752, upper bound: 2.3087073
time: 5.80 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3449753, upper bound: 2.3113494
time: 5.50 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -6.1852236, -1.7829347, -6.1962519, -1.7841630, -4.4010606, 4.4133172
1: -15.2631798, -10.7432575, -15.2972393, -10.7122097, -4.5509701, 4.5539818
2: -9.0590172, -4.6691799, -9.1495333, -4.5761361, -4.3761625, 4.3848257
3: -7.5695829, -3.6011109, -7.5837040, -3.5838339, -3.9857490, 3.9825931
4: -12.2054796, -7.3996406, -12.2653751, -7.3659964, -4.8394833, 4.8657346
5: -5.9826055, -2.2345085, -6.0167756, -2.2000248, -3.7825806, 3.7822671
6: -13.7674370, -9.3107300, -13.7735767, -9.2978630, -4.1665230, 4.1465082
7: -10.2198896, -5.8791976, -10.2340193, -5.8906541, -4.3292356, 4.3548217
8: 7.8692045, 11.0702295, 7.8268309, 11.1009655, -3.2317610, 3.2433987
9: -7.1358552, -3.2471225, -7.1479235, -3.2283878, -3.8630948, 3.7682900

Time for backsubstitution: 13.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454208, upper bound: 2.3221582
time: 5.85 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454207, upper bound: 2.3247353
time: 5.89 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1710052, -1.8091354, -6.2133961, -1.7566996, -4.4143057, 4.4042606
1: -15.2505445, -10.7507181, -15.3121748, -10.7039547, -4.5465899, 4.5614567
2: -9.0504265, -4.6823287, -9.1586056, -4.5603304, -4.3888659, 4.3794813
3: -7.5379372, -3.6306880, -7.6162853, -3.5472577, -3.9906795, 3.9855974
4: -12.1911879, -7.4107089, -12.2817230, -7.3541961, -4.8369918, 4.8710141
5: -5.9681535, -2.2458134, -6.0326071, -2.1862943, -3.7818592, 3.7867937
6: -13.7295284, -9.3505440, -13.8142920, -9.2481222, -4.1767740, 4.1364951
7: -10.1899185, -5.9047527, -10.2709503, -5.8640852, -4.3258333, 4.3661976
8: 7.8826990, 11.0574713, 7.8114200, 11.1164789, -3.2337799, 3.2460513
9: -7.1110320, -3.2506745, -7.1753473, -3.2234244, -3.8406634, 3.7827396

Time for backsubstitution: 13.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3319927, upper bound: 2.3216824
time: 5.44 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454155, upper bound: 2.3221589
time: 5.11 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1879406, -1.7817001, -6.2133961, -1.7566996, -4.4312410, 4.4316959
1: -15.2639418, -10.7425203, -15.3121748, -10.7039547, -4.5599871, 4.5696545
2: -9.0594826, -4.6674356, -9.1586056, -4.5603304, -4.3997073, 4.3963776
3: -7.5704584, -3.5940981, -7.6162853, -3.5472577, -4.0232010, 4.0221872
4: -12.2075644, -7.3990145, -12.2817230, -7.3541961, -4.8533683, 4.8827085
5: -5.9838691, -2.2321656, -6.0326071, -2.1862943, -3.7975748, 3.8004415
6: -13.7705040, -9.3012981, -13.8142920, -9.2481222, -4.1873226, 4.1702876
7: -10.2259922, -5.8782072, -10.2709503, -5.8640852, -4.3619070, 4.3927431
8: 7.8683772, 11.0728416, 7.8114200, 11.1164789, -3.2481017, 3.2614217
9: -7.1385517, -3.2466414, -7.1753473, -3.2234244, -3.8778486, 3.7894669

Time for backsubstitution: 13.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3319927, upper bound: 2.3242783
time: 5.48 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3454155, upper bound: 2.3247359
time: 6.26 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.1362791, -1.8214746, -6.1736069, -1.7850957, -4.3511834, 4.3521323
1: -15.2382622, -10.7401867, -15.2520123, -10.7452097, -4.4930525, 4.5118256
2: -9.1377182, -4.6134310, -9.0580978, -4.6760082, -4.3636951, 4.3397889
3: -7.5652986, -3.5954003, -7.5674992, -3.6029730, -3.9623256, 3.9720988
4: -12.2253914, -7.3918042, -12.2015228, -7.4047403, -4.8206511, 4.8097186
5: -5.9956317, -2.2169721, -5.9800134, -2.2374291, -3.7582026, 3.7630413
6: -13.7303085, -9.3233337, -13.7589760, -9.3125343, -4.1020031, 4.1323514
7: -10.2096767, -5.9124813, -10.2175169, -5.8833332, -4.3263435, 4.3050356
8: 7.8561664, 11.0790901, 7.8747721, 11.0684881, -3.2123218, 3.2043180
9: -7.1246834, -3.2466125, -7.1318150, -3.2500587, -3.7266493, 3.8404260

Time for backsubstitution: 13.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3087069, upper bound: 2.3449744
time: 6.11 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3087068, upper bound: 2.3449759
time: 14.93 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -6.1962519, -1.7841630, -6.1852236, -1.7829347, -4.4133172, 4.4010606
1: -15.2972393, -10.7122097, -15.2631798, -10.7432575, -4.5539818, 4.5509701
2: -9.1495333, -4.5761361, -9.0590172, -4.6691799, -4.3848257, 4.3761625
3: -7.5837040, -3.5838339, -7.5695829, -3.6011109, -3.9825931, 3.9857490
4: -12.2653751, -7.3659964, -12.2054796, -7.3996406, -4.8657346, 4.8394833
5: -6.0167756, -2.2000248, -5.9826055, -2.2345085, -3.7822671, 3.7825806
6: -13.7735767, -9.2978630, -13.7674370, -9.3107300, -4.1465082, 4.1665230
7: -10.2340193, -5.8906541, -10.2198896, -5.8791976, -4.3548217, 4.3292356
8: 7.8268309, 11.1009655, 7.8692045, 11.0702295, -3.2433987, 3.2317610
9: -7.1479235, -3.2283878, -7.1358552, -3.2471225, -3.7682900, 3.8630939

Time for backsubstitution: 13.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221578, upper bound: 2.3454214
time: 5.25 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221578, upper bound: 2.3454217
time: 4.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.2133961, -1.7566996, -6.1710052, -1.8091354, -4.4042606, 4.4143057
1: -15.3121748, -10.7039547, -15.2505445, -10.7507181, -4.5614567, 4.5465899
2: -9.1586056, -4.5603304, -9.0504265, -4.6823287, -4.3794813, 4.3888659
3: -7.6162853, -3.5472577, -7.5379372, -3.6306880, -3.9855974, 3.9906795
4: -12.2817230, -7.3541961, -12.1911879, -7.4107089, -4.8710141, 4.8369918
5: -6.0326071, -2.1862943, -5.9681535, -2.2458134, -3.7867937, 3.7818592
6: -13.8142920, -9.2481222, -13.7295284, -9.3505440, -4.1364951, 4.1767740
7: -10.2709503, -5.8640852, -10.1899185, -5.9047527, -4.3661976, 4.3258333
8: 7.8114200, 11.1164789, 7.8826990, 11.0574713, -3.2460513, 3.2337799
9: -7.1753473, -3.2234244, -7.1110320, -3.2506745, -3.7827387, 3.8406634

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216816, upper bound: 2.3346068
time: 5.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221583, upper bound: 2.3479835
time: 5.25 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.2133961, -1.7566996, -6.1879406, -1.7817001, -4.4316959, 4.4312410
1: -15.3121748, -10.7039547, -15.2639418, -10.7425203, -4.5696545, 4.5599871
2: -9.1586056, -4.5603304, -9.0594826, -4.6674356, -4.3963776, 4.3997078
3: -7.6162853, -3.5472577, -7.5704584, -3.5940981, -4.0221872, 4.0232010
4: -12.2817230, -7.3541961, -12.2075644, -7.3990145, -4.8827085, 4.8533683
5: -6.0326071, -2.1862943, -5.9838691, -2.2321656, -3.8004415, 3.7975748
6: -13.8142920, -9.2481222, -13.7705040, -9.3012981, -4.1702881, 4.1873226
7: -10.2709503, -5.8640852, -10.2259922, -5.8782072, -4.3927431, 4.3619070
8: 7.8114200, 11.1164789, 7.8683772, 11.0728416, -3.2614217, 3.2481017
9: -7.1753473, -3.2234244, -7.1385517, -3.2466414, -3.7894669, 3.8778486

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216817, upper bound: 2.3346067
time: 4.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221583, upper bound: 2.3479838
time: 5.13 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1962962, -1.7841535, -6.1962962, -1.7841535, -4.4121428, 4.4121428
1: -15.2972927, -10.7122021, -15.2972927, -10.7122021, -4.5850906, 4.5850906
2: -9.1495380, -4.5761046, -9.1495380, -4.5761046, -4.4612894, 4.4612899
3: -7.5837145, -3.5838249, -7.5837145, -3.5838249, -3.9998896, 3.9998896
4: -12.2653923, -7.3659816, -12.2653923, -7.3659816, -4.8994107, 4.8994107
5: -6.0167847, -2.2000177, -6.0167847, -2.2000177, -3.8167670, 3.8167670
6: -13.7736197, -9.2978554, -13.7736197, -9.2978554, -4.1475534, 4.1475534
7: -10.2340288, -5.8906298, -10.2340288, -5.8906298, -4.3433990, 4.3433990
8: 7.8268032, 11.1009731, 7.8268032, 11.1009731, -3.2741699, 3.2741699
9: -7.1479387, -3.2283797, -7.1479387, -3.2283797, -3.8958826, 3.8958826

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3087081, upper bound: 2.3449724
time: 5.19 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221591, upper bound: 2.3454171
time: 7.89 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2133961, -1.7566996, -6.1962962, -1.7841535, -4.4292426, 4.4395967
1: -15.3121748, -10.7039547, -15.2972927, -10.7122021, -4.5999727, 4.5933380
2: -9.1586056, -4.5603304, -9.1495380, -4.5761046, -4.4663277, 4.4771671
3: -7.6162853, -3.5472577, -7.5837145, -3.5838249, -4.0324602, 4.0364571
4: -12.2817230, -7.3541961, -12.2653923, -7.3659816, -4.9157414, 4.9111962
5: -6.0326071, -2.1862943, -6.0167847, -2.2000177, -3.8325894, 3.8304904
6: -13.8142920, -9.2481222, -13.7736197, -9.2978554, -4.1896782, 4.1969876
7: -10.2709503, -5.8640852, -10.2340288, -5.8906298, -4.3803205, 4.3699436
8: 7.8114200, 11.1164789, 7.8268032, 11.1009731, -3.2895532, 3.2896757
9: -7.1753473, -3.2234244, -7.1479387, -3.2283797, -3.9259577, 3.9025598

Time for backsubstitution: 13.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3216829, upper bound: 2.3346071
time: 7.20 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221595, upper bound: 2.3479842
time: 6.50 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1962962, -1.7841535, -6.2133961, -1.7566996, -4.4395967, 4.4292426
1: -15.2972927, -10.7122021, -15.3121748, -10.7039547, -4.5933380, 4.5999727
2: -9.1495380, -4.5761046, -9.1586056, -4.5603304, -4.4771667, 4.4663272
3: -7.5837145, -3.5838249, -7.6162853, -3.5472577, -4.0364571, 4.0324602
4: -12.2653923, -7.3659816, -12.2817230, -7.3541961, -4.9111962, 4.9157414
5: -6.0167847, -2.2000177, -6.0326071, -2.1862943, -3.8304904, 3.8325894
6: -13.7736197, -9.2978554, -13.8142920, -9.2481222, -4.1969881, 4.1896777
7: -10.2340288, -5.8906298, -10.2709503, -5.8640852, -4.3699436, 4.3803205
8: 7.8268032, 11.1009731, 7.8114200, 11.1164789, -3.2896757, 3.2895532
9: -7.1479387, -3.2283797, -7.1753473, -3.2234244, -3.9025593, 3.9259577

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3087034, upper bound: 2.3449724
time: 5.43 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.3221543, upper bound: 2.3454152
time: 6.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.86 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3216868, upper bound: 2.3087074
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3216867, upper bound: 2.3113501
IS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221635, upper bound: 2.3221587
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221634, upper bound: 2.3247358
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3087072, upper bound: 2.3216813
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221582, upper bound: 2.3221595
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3216819, upper bound: 2.3113496
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221588, upper bound: 2.3247355
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3449752, upper bound: 2.3087073
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3449753, upper bound: 2.3113494
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3454208, upper bound: 2.3221582
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3454207, upper bound: 2.3247353
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3319927, upper bound: 2.3216824
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3454155, upper bound: 2.3221589
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3319927, upper bound: 2.3242783
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3454155, upper bound: 2.3247359
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3087069, upper bound: 2.3449744
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3087068, upper bound: 2.3449759
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221578, upper bound: 2.3454214
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221578, upper bound: 2.3454217
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3216816, upper bound: 2.3346068
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221583, upper bound: 2.3479835
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3216817, upper bound: 2.3346067
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221583, upper bound: 2.3479838
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3087081, upper bound: 2.3449724
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221591, upper bound: 2.3454171
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3216829, upper bound: 2.3346071
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221595, upper bound: 2.3479842
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3087034, upper bound: 2.3449724
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.86
Output dim: 8, lower bound: -2.3221543, upper bound: 2.3454152
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -2.3247531, upper bound: 2.3480050
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=3.3050923347473145
rel_dist={8: [-2.3480243628211177, 2.348024799945545]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186595, upper bound: 2.1376545
time: 5.15 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376531, upper bound: 2.1376539
time: 6.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.79 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 11.79
Output dim: 8, lower bound: -2.1186595, upper bound: 2.1376545
IS_B2, status: Status.UNKNOWN, split count: 1, time: 11.79
Output dim: 8, lower bound: -2.1376531, upper bound: 2.1376539

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -6.2091494, -1.7597084, -6.1879435, -1.7816978, -4.4256983, 4.4279652
1: -15.3010950, -10.7063179, -15.2639427, -10.7425175, -4.5300636, 4.4406815
2: -9.1324720, -4.5660038, -9.0594845, -4.6674333, -4.1667862, 4.2481713
3: -7.6057081, -3.5499494, -7.5704598, -3.5940862, -4.0116220, 4.0205107
4: -12.2760239, -7.3655386, -12.2075682, -7.3990154, -4.8770084, 4.8420296
5: -6.0290804, -2.1972632, -5.9838700, -2.2321615, -3.6720533, 3.6643038
6: -13.8111706, -9.2609310, -13.7705059, -9.3012829, -3.9632015, 3.9703932
7: -10.2642660, -5.8656473, -10.2260075, -5.8782048, -4.3860612, 4.3603601
8: 7.8167686, 11.1055670, 7.8683748, 11.0728474, -3.1755972, 3.0677624
9: -7.1719365, -3.2289586, -7.1385541, -3.2466414, -3.5830760, 3.7048755

Time for backsubstitution: 13.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1186580
time: 4.94 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1376547
time: 5.06 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -6.2134089, -1.7566910, -6.2133999, -1.7566977, -4.4567113, 4.4535165
1: -15.3121939, -10.7039461, -15.3121786, -10.7039528, -4.6050930, 4.5886168
2: -9.1586838, -4.5603180, -9.1586056, -4.5603261, -4.3011179, 4.3024845
3: -7.6163106, -3.5472407, -7.6162882, -3.5472469, -4.0690637, 4.0477238
4: -12.2817440, -7.3541565, -12.2817307, -7.3541937, -4.9275503, 4.9275742
5: -6.0326147, -2.1862621, -6.0326066, -2.1862900, -3.7146559, 3.7216382
6: -13.8143024, -9.2480803, -13.8142948, -9.2481070, -4.0003033, 4.0281916
7: -10.2709703, -5.8640804, -10.2709599, -5.8640842, -4.4068861, 4.4068794
8: 7.8114080, 11.1164961, 7.8114176, 11.1164827, -3.2062726, 3.2178319
9: -7.1753602, -3.2234144, -7.1753492, -3.2234240, -3.7588367, 3.7479610

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1355017, upper bound: 2.1376512
time: 5.55 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1376504, upper bound: 2.1376513
time: 7.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.63 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 25.63
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1186580
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 25.63
Output dim: 8, lower bound: -2.1186570, upper bound: 2.1376547
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 25.63
Output dim: 8, lower bound: -2.1355017, upper bound: 2.1376512
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 25.63
Output dim: 8, lower bound: -2.1376504, upper bound: 2.1376513

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 13.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186567, upper bound: 2.1164274
time: 5.16 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186567, upper bound: 2.1186550
time: 5.34 seconds

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186567, upper bound: 2.1355031
time: 5.35 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186568, upper bound: 2.1376517
time: 5.34 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -6.2093792, -1.7585125, -6.1962962, -1.7841535, -4.4252257, 4.4307814
1: -15.3110323, -10.7050209, -15.2972927, -10.7122021, -4.5817308, 4.5664124
2: -9.1579847, -4.5629034, -9.1495380, -4.5761046, -4.2822180, 4.2886124
3: -7.6150231, -3.5575268, -7.5837145, -3.5838249, -4.0311985, 4.0023775
4: -12.2787018, -7.3550911, -12.2653923, -7.3659816, -4.9127202, 4.9103012
5: -6.0307775, -2.1897221, -6.0167847, -2.2000177, -3.6987410, 3.6999073
6: -13.8098745, -9.2619143, -13.7736197, -9.2978554, -3.9463739, 3.9697208
7: -10.2620173, -5.8655224, -10.2340288, -5.8906298, -4.3713875, 4.3685064
8: 7.8126469, 11.1126280, 7.8268032, 11.1009731, -3.1893196, 3.1963382
9: -7.1714535, -3.2240953, -7.1479387, -3.2283797, -3.7462978, 3.7149358

Time for backsubstitution: 13.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164262, upper bound: 2.1186532
time: 9.12 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164263, upper bound: 2.1376508
time: 6.32 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -6.2134066, -1.7566929, -6.2133961, -1.7566996, -4.4567070, 4.4528275
1: -15.3121958, -10.7039471, -15.3121748, -10.7039547, -4.5985136, 4.5864882
2: -9.1586847, -4.5603218, -9.1586056, -4.5603304, -4.2977996, 4.3053551
3: -7.6163101, -3.5472417, -7.6162853, -3.5472577, -4.0511770, 4.0477209
4: -12.2817402, -7.3541555, -12.2817230, -7.3541961, -4.9275441, 4.9275675
5: -6.0326152, -2.1862602, -6.0326071, -2.1862943, -3.7145529, 3.7241330
6: -13.8143005, -9.2480803, -13.8142920, -9.2481222, -3.9643192, 4.0121408
7: -10.2709694, -5.8640804, -10.2709503, -5.8640852, -4.4068842, 4.4068699
8: 7.8114090, 11.1164932, 7.8114200, 11.1164789, -3.2118673, 3.2069292
9: -7.1753578, -3.2234144, -7.1753473, -3.2234244, -3.7569513, 3.7508183

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186542, upper bound: 2.1186545
time: 9.22 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186542, upper bound: 2.1376511
time: 7.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.79 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1186567, upper bound: 2.1164274
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1186567, upper bound: 2.1186550
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1186567, upper bound: 2.1355031
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1186568, upper bound: 2.1376517
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1164262, upper bound: 2.1186532
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1164263, upper bound: 2.1376508
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1186542, upper bound: 2.1186545
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.79
Output dim: 8, lower bound: -2.1186542, upper bound: 2.1376511

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.1710052, -1.8091354, -6.1839600, -1.7835202, -4.3681450, 4.3571949
1: -15.2505445, -10.7507181, -15.2628183, -10.7436008, -4.3612814, 4.3597498
2: -9.0504265, -4.6823287, -9.0587978, -4.6700077, -4.0793056, 4.0746756
3: -7.5379372, -3.6306880, -7.5691710, -3.6043844, -3.9335527, 3.9384830
4: -12.1911879, -7.4107089, -12.2045078, -7.3999352, -4.7912526, 4.7937989
5: -5.9681535, -2.2458134, -5.9820113, -2.2355967, -3.6073589, 3.6134748
6: -13.7295284, -9.3505440, -13.7659988, -9.3151293, -3.8653145, 3.8676872
7: -10.1899185, -5.9047527, -10.2170391, -5.8796611, -4.3102574, 4.3122864
8: 7.8826990, 11.0574713, 7.8695922, 11.0690155, -3.0139017, 3.0176721
9: -7.1110320, -3.2506745, -7.1345882, -3.2473438, -3.5144134, 3.5349765

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1147983
time: 8.11 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1164151
time: 5.10 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -6.1879406, -1.7817001, -6.1879425, -1.7816992, -4.3900986, 4.3903170
1: -15.2639418, -10.7425203, -15.2639427, -10.7425184, -4.3797951, 4.3795609
2: -9.0594826, -4.6674356, -9.0594835, -4.6674328, -4.0913792, 4.0913396
3: -7.5704584, -3.5940981, -7.5704589, -3.5940890, -3.9763694, 3.9530964
4: -12.2075644, -7.3990145, -12.2075701, -7.3990154, -4.8085489, 4.8085556
5: -5.9838691, -2.2321656, -5.9838686, -2.2321613, -3.6325989, 3.6291571
6: -13.7705040, -9.3012981, -13.7705059, -9.3012848, -3.9211025, 3.8831091
7: -10.2259922, -5.8782072, -10.2260056, -5.8782043, -4.3477879, 4.3477983
8: 7.8683772, 11.0728416, 7.8683748, 11.0728455, -3.0317101, 3.0346656
9: -7.1385517, -3.2466414, -7.1385536, -3.2466412, -3.5458021, 3.5457654

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1170273
time: 5.34 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1186428
time: 5.40 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.1962962, -1.7841535, -6.1839600, -1.7835202, -4.3946285, 4.3808575
1: -15.2972927, -10.7122021, -15.2628183, -10.7436008, -4.5176420, 4.4017472
2: -9.1495380, -4.5761046, -9.0587978, -4.6700077, -4.1720009, 4.1824036
3: -7.5837145, -3.5838249, -7.5691710, -3.6043844, -3.9793301, 3.9853461
4: -12.2653923, -7.3659816, -12.2045078, -7.3999352, -4.8654571, 4.8385262
5: -6.0167847, -2.2000177, -5.9820113, -2.2355967, -3.6536131, 3.6595240
6: -13.7736197, -9.2978554, -13.7659988, -9.3151293, -3.9034338, 3.9295282
7: -10.2340288, -5.8906298, -10.2170391, -5.8796611, -4.3543677, 4.3264093
8: 7.8268032, 11.1009731, 7.8695922, 11.0690155, -3.1494088, 3.0632820
9: -7.1479387, -3.2283797, -7.1345882, -3.2473438, -3.5542884, 3.6870599

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068657, upper bound: 2.1339273
time: 6.98 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186411, upper bound: 2.1354868
time: 5.19 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.2133961, -1.7566996, -6.1879425, -1.7816992, -4.4166279, 4.4140043
1: -15.3121748, -10.7039547, -15.2639427, -10.7425184, -4.5377216, 4.4215751
2: -9.1586056, -4.5603304, -9.0594835, -4.6674328, -4.1792297, 4.1979709
3: -7.6162853, -3.5472577, -7.5704589, -3.5940890, -4.0221963, 4.0011415
4: -12.2817230, -7.3541961, -12.2075701, -7.3990154, -4.8827076, 4.8533740
5: -6.0326071, -2.1862943, -5.9838686, -2.2321613, -3.6777716, 3.6752968
6: -13.8142920, -9.2481222, -13.7705059, -9.3012848, -3.9460306, 3.9475403
7: -10.2709503, -5.8640852, -10.2260056, -5.8782043, -4.3927460, 4.3619204
8: 7.8114200, 11.1164789, 7.8683748, 11.0728455, -3.1598792, 3.0802219
9: -7.1753473, -3.2234244, -7.1385536, -3.2466412, -3.5856447, 3.6978426

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4555

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1068658, upper bound: 2.1361182
time: 8.39 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186412, upper bound: 2.1376355
time: 5.04 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1839600, -1.7835202, -6.1962962, -1.7841535, -4.3808575, 4.3946285
1: -15.2628183, -10.7436008, -15.2972927, -10.7122021, -4.4017467, 4.5176425
2: -9.0587978, -4.6700077, -9.1495380, -4.5761046, -4.1824045, 4.1720014
3: -7.5691710, -3.6043844, -7.5837145, -3.5838249, -3.9853461, 3.9793301
4: -12.2045078, -7.3999352, -12.2653923, -7.3659816, -4.8385262, 4.8654571
5: -5.9820113, -2.2355967, -6.0167847, -2.2000177, -3.6595240, 3.6536131
6: -13.7659988, -9.3151293, -13.7736197, -9.2978554, -3.9295282, 3.9034333
7: -10.2170391, -5.8796611, -10.2340288, -5.8906298, -4.3264093, 4.3543677
8: 7.8695922, 11.0690155, 7.8268032, 11.1009731, -3.0632820, 3.1494091
9: -7.1345882, -3.2473438, -7.1479387, -3.2283797, -3.6870599, 3.5542889

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1147956, upper bound: 2.1068631
time: 5.86 seconds

## Relational analysis of IS_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1186382
time: 11.93 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2093716, -1.7585206, -6.1962962, -1.7841535, -4.4252181, 4.4377756
1: -15.3110161, -10.7050266, -15.2972927, -10.7122021, -4.5816956, 4.5828552
2: -9.1579075, -4.5629115, -9.1495380, -4.5761046, -4.2648554, 4.2699256
3: -7.6150007, -3.5575318, -7.5837145, -3.5838249, -4.0083656, 4.0023704
4: -12.2786884, -7.3551302, -12.2653923, -7.3659816, -4.9127069, 4.9102621
5: -6.0307684, -2.1897492, -6.0167847, -2.2000177, -3.6987324, 3.6929207
6: -13.8098650, -9.2619390, -13.7736197, -9.2978554, -3.9463663, 3.9447470
7: -10.2620096, -5.8655248, -10.2340288, -5.8906298, -4.3713799, 4.3685040
8: 7.8126535, 11.1126156, 7.8268032, 11.1009731, -3.1893120, 3.1848631
9: -7.1714458, -3.2241068, -7.1479387, -3.2283797, -3.7462816, 3.7256546

Time for backsubstitution: 13.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_B2_B1_A2_B1

### Relational analysis result of IS_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1147955, upper bound: 2.1259576
time: 5.24 seconds

## Relational analysis of IS_B2_B1_A2_B2

### Relational analysis result of IS_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1376353
time: 7.66 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1879425, -1.7816992, -6.2133961, -1.7566996, -4.4140043, 4.4166279
1: -15.2639427, -10.7425184, -15.3121748, -10.7039547, -4.4215755, 4.5377216
2: -9.0594835, -4.6674328, -9.1586056, -4.5603304, -4.1979709, 4.1792293
3: -7.5704589, -3.5940890, -7.6162853, -3.5472577, -4.0011415, 4.0221963
4: -12.2075701, -7.3990154, -12.2817230, -7.3541961, -4.8533740, 4.8827076
5: -5.9838686, -2.2321613, -6.0326071, -2.1862943, -3.6752968, 3.6777716
6: -13.7705059, -9.3012848, -13.8142920, -9.2481222, -3.9475403, 3.9460311
7: -10.2260056, -5.8782043, -10.2709503, -5.8640852, -4.3619204, 4.3927460
8: 7.8683748, 11.0728455, 7.8114200, 11.1164789, -3.0802219, 3.1598797
9: -7.1385536, -3.2466412, -7.1753473, -3.2234244, -3.6978431, 3.5856447

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4555

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1170239, upper bound: 2.1068630
time: 21.64 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1186392, upper bound: 2.1186379
time: 10.33 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2133980, -1.7566981, -6.2133961, -1.7566996, -4.4566984, 4.4566979
1: -15.3121758, -10.7039528, -15.3121748, -10.7039547, -4.5984964, 4.5973692
2: -9.1586065, -4.5603285, -9.1586056, -4.5603304, -4.2804360, 4.2801223
3: -7.6162877, -3.5472491, -7.6162853, -3.5472577, -4.0167894, 4.0477128
4: -12.2817287, -7.3541965, -12.2817230, -7.3541961, -4.9275327, 4.9275265
5: -6.0326085, -2.1862922, -6.0326071, -2.1862943, -3.7145462, 3.7173247
6: -13.8142958, -9.2481089, -13.8142920, -9.2481222, -3.9643106, 4.0002904
7: -10.2709579, -5.8640847, -10.2709503, -5.8640852, -4.4068727, 4.4068656
8: 7.8114166, 11.1164818, 7.8114200, 11.1164789, -3.2094922, 3.1998775
9: -7.1753502, -3.2234244, -7.1753473, -3.2234244, -3.7569342, 3.7613602

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164220, upper bound: 2.1354980
time: 6.72 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164220, upper bound: 2.1376512
time: 8.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.40 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1147983
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1164151
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1068661, upper bound: 2.1170273
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1186415, upper bound: 2.1186428
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1068657, upper bound: 2.1339273
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1186411, upper bound: 2.1354868
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1068658, upper bound: 2.1361182
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1186412, upper bound: 2.1376355
IS_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1147956, upper bound: 2.1068631
IS_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1186382
IS_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1147955, upper bound: 2.1259576
IS_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1376353
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1170239, upper bound: 2.1068630
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1186392, upper bound: 2.1186379
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1164220, upper bound: 2.1354980
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.40
Output dim: 8, lower bound: -2.1164220, upper bound: 2.1376512

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -6.1108990, -1.8464925, -6.1670551, -1.7866788, -4.3068399, 4.3057251
1: -15.1916485, -10.7787066, -15.2465677, -10.7464695, -4.2981453, 4.3124595
2: -9.0386009, -4.7194743, -9.0574636, -4.6799469, -4.0543966, 4.0295982
3: -7.5195389, -3.6422558, -7.5661378, -3.6070938, -3.9124451, 3.9238820
4: -12.1512642, -7.4365139, -12.1987295, -7.4073510, -4.7439132, 4.7622156
5: -5.9469881, -2.2628121, -5.9782200, -2.2398388, -3.5751266, 3.5894928
6: -13.6862612, -9.3759060, -13.7536831, -9.3177795, -3.8184423, 3.8299541
7: -10.1658173, -5.9266062, -10.2135878, -5.8856831, -4.2801342, 4.2869816
8: 7.9118772, 11.0355921, 7.8777013, 11.0664644, -2.9817820, 2.9871531
9: -7.0880852, -3.2687817, -7.1286869, -3.2516174, -3.4809642, 3.5109792

Time for backsubstitution: 13.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1147988
time: 5.05 seconds

## Relational analysis of IS_B1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1147976
time: 5.28 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -6.1709595, -1.8091455, -6.1839566, -1.7835217, -4.3653545, 4.3571825
1: -15.2504911, -10.7507257, -15.2628155, -10.7436018, -4.3611212, 4.3637357
2: -9.0504208, -4.6823602, -9.0587969, -4.6700068, -4.0792980, 4.0659847
3: -7.5379286, -3.6306965, -7.5691686, -3.6043870, -3.9335415, 3.9384720
4: -12.1911669, -7.4107237, -12.2045078, -7.3999367, -4.7912302, 4.7937841
5: -5.9681416, -2.2458189, -5.9820123, -2.2355981, -3.6229782, 3.6130233
6: -13.7294884, -9.3505507, -13.7659960, -9.3151302, -3.8632469, 3.8676772
7: -10.1899071, -5.9047737, -10.2170382, -5.8796635, -4.3102436, 4.3122644
8: 7.8827248, 11.0574627, 7.8695936, 11.0690136, -3.0043206, 3.0176625
9: -7.1110163, -3.2506840, -7.1345873, -3.2473459, -3.5236979, 3.5346961

Time for backsubstitution: 13.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164090, upper bound: 2.1164157
time: 6.21 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164089, upper bound: 2.1164139
time: 7.45 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -6.1277561, -1.8190875, -6.1710329, -1.7848697, -4.3287487, 4.3388529
1: -15.2049675, -10.7705107, -15.2476873, -10.7453880, -4.3165636, 4.3322682
2: -9.0476589, -4.7045970, -9.0581474, -4.6773758, -4.0664883, 4.0462236
3: -7.5520773, -3.6056828, -7.5674276, -3.5967989, -3.9552784, 3.9374628
4: -12.1676655, -7.4248295, -12.2017918, -7.4064302, -4.7612352, 4.7769623
5: -5.9627142, -2.2491870, -5.9800744, -2.2364011, -3.6003780, 3.6051931
6: -13.7271729, -9.3266525, -13.7581854, -9.3039341, -3.8740644, 3.8453732
7: -10.2019262, -5.9000616, -10.2225561, -5.8842278, -4.3176985, 4.3224945
8: 7.8975549, 11.0509739, 7.8764839, 11.0702953, -2.9995947, 3.0041671
9: -7.1151900, -3.2647576, -7.1326485, -3.2509148, -3.5122638, 3.5217514

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1170273
time: 5.35 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1170258
time: 6.39 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -6.1878924, -1.7817097, -6.1879392, -1.7817011, -4.3873100, 4.3903055
1: -15.2638922, -10.7425289, -15.2639389, -10.7425194, -4.3796377, 4.3835464
2: -9.0594749, -4.6674690, -9.0594826, -4.6674361, -4.0913734, 4.0826516
3: -7.5704484, -3.5941064, -7.5704589, -3.5940895, -3.9763589, 3.9518657
4: -12.2075491, -7.3990283, -12.2075672, -7.3990169, -4.8085322, 4.8085389
5: -5.9838567, -2.2321744, -5.9838691, -2.2321620, -3.6482229, 3.6287050
6: -13.7704611, -9.3013020, -13.7705021, -9.3012848, -3.9190369, 3.8831000
7: -10.2259874, -5.8782277, -10.2260056, -5.8782077, -4.3477798, 4.3477778
8: 7.8684015, 11.0728359, 7.8683767, 11.0728436, -3.0221291, 3.0346560
9: -7.1385341, -3.2466507, -7.1385517, -3.2466426, -3.5550876, 3.5454860

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164089, upper bound: 2.1186438
time: 6.09 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164090, upper bound: 2.1186418
time: 6.64 seconds

## BFS IS instance: IS_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -6.1362791, -1.8214746, -6.1670551, -1.7866788, -4.3333635, 4.3294210
1: -15.2382622, -10.7401867, -15.2465677, -10.7464695, -4.4550114, 4.3544602
2: -9.1377182, -4.6134310, -9.0574636, -4.6799469, -4.1428938, 4.1375971
3: -7.5652986, -3.5954003, -7.5661378, -3.6070938, -3.9582047, 3.9707375
4: -12.2253914, -7.3918042, -12.1987295, -7.4073510, -4.8180404, 4.8069253
5: -5.9956317, -2.2169721, -5.9782200, -2.2398388, -3.6213388, 3.6355681
6: -13.7303085, -9.3233337, -13.7536831, -9.3177795, -3.8564315, 3.8912501
7: -10.2096767, -5.9124813, -10.2135878, -5.8856831, -4.3239937, 4.3011065
8: 7.8561664, 11.0790901, 7.8777013, 11.0664644, -3.1169143, 3.0313358
9: -7.1246834, -3.2466125, -7.1286869, -3.2516174, -3.5207400, 3.6624165

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046324, upper bound: 2.1339278
time: 5.33 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046324, upper bound: 2.1339263
time: 5.26 seconds

## BFS IS instance: IS_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -6.1962519, -1.7841630, -6.1839566, -1.7835217, -4.3918371, 4.3808451
1: -15.2972393, -10.7122097, -15.2628155, -10.7436018, -4.5174923, 4.4057317
2: -9.1495333, -4.5761361, -9.0587969, -4.6700068, -4.1678128, 4.1734180
3: -7.5837040, -3.5838339, -7.5691686, -3.6043870, -3.9793169, 3.9853346
4: -12.2653751, -7.3659964, -12.2045078, -7.3999367, -4.8654385, 4.8385115
5: -6.0167756, -2.2000248, -5.9820123, -2.2355981, -3.6667767, 3.6590719
6: -13.7735767, -9.2978630, -13.7659960, -9.3151302, -3.9004211, 3.9295177
7: -10.2340193, -5.8906541, -10.2170382, -5.8796635, -4.3543558, 4.3263841
8: 7.8268309, 11.1009655, 7.8695936, 11.0690136, -3.1388865, 3.0579929
9: -7.1479235, -3.2283878, -7.1345873, -3.2473459, -3.5635738, 3.6867986

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164085, upper bound: 2.1354880
time: 5.77 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164086, upper bound: 2.1354857
time: 6.43 seconds

## BFS IS instance: IS_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -6.1533027, -1.7940540, -6.1710329, -1.7848697, -4.3553324, 4.3625755
1: -15.2530632, -10.7319393, -15.2476873, -10.7453880, -4.4749737, 4.3742781
2: -9.1467915, -4.5976763, -9.0581474, -4.6773758, -4.1501269, 4.1530409
3: -7.5978894, -3.5588515, -7.5674276, -3.5967989, -4.0010905, 3.9854574
4: -12.2417507, -7.3800268, -12.2017918, -7.4064302, -4.8353205, 4.8217649
5: -6.0114608, -2.2032707, -5.9800744, -2.2364011, -3.6455755, 3.6513538
6: -13.7709198, -9.2735939, -13.7581854, -9.3039341, -3.8989563, 3.9092579
7: -10.2466278, -5.8859382, -10.2225561, -5.8842278, -4.3624001, 4.3366179
8: 7.8407817, 11.0946217, 7.8764839, 11.0702953, -3.1273913, 3.0435085
9: -7.1516843, -3.2416608, -7.1326485, -3.2509148, -3.5519452, 3.6731858

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046324, upper bound: 2.1361181
time: 5.63 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1046325, upper bound: 2.1361169
time: 5.90 seconds

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -6.2133484, -1.7567105, -6.1879392, -1.7817011, -4.4138393, 4.4139919
1: -15.3121214, -10.7039604, -15.2639389, -10.7425194, -4.5375738, 4.4208622
2: -9.1586027, -4.5603623, -9.0594826, -4.6674361, -4.1750402, 4.1889868
3: -7.6162753, -3.5472648, -7.5704589, -3.5940895, -4.0221858, 3.9999113
4: -12.2817097, -7.3542099, -12.2075672, -7.3990169, -4.8826928, 4.8533573
5: -6.0325961, -2.1863008, -5.9838691, -2.2321620, -3.6851664, 3.6748457
6: -13.8142519, -9.2481308, -13.7705021, -9.3012848, -3.9430199, 3.9475284
7: -10.2709389, -5.8641076, -10.2260056, -5.8782077, -4.3927312, 4.3618979
8: 7.8114467, 11.1164694, 7.8683767, 11.0728436, -3.1493573, 3.0701566
9: -7.1753292, -3.2234340, -7.1385517, -3.2466426, -3.5949292, 3.6975842

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164086, upper bound: 2.1376352
time: 5.76 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164086, upper bound: 2.1376346
time: 6.76 seconds

## BFS IS instance: IS_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1670551, -1.7866788, -6.1362791, -1.8214746, -4.3294210, 4.3333635
1: -15.2465677, -10.7464695, -15.2382622, -10.7401867, -4.3544598, 4.4550114
2: -9.0574636, -4.6799469, -9.1377182, -4.6134310, -4.1375971, 4.1428943
3: -7.5661378, -3.6070938, -7.5652986, -3.5954003, -3.9707375, 3.9582047
4: -12.1987295, -7.4073510, -12.2253914, -7.3918042, -4.8069253, 4.8180404
5: -5.9782200, -2.2398388, -5.9956317, -2.2169721, -3.6355677, 3.6213388
6: -13.7536831, -9.3177795, -13.7303085, -9.3233337, -3.8912497, 3.8564320
7: -10.2135878, -5.8856831, -10.2096767, -5.9124813, -4.3011065, 4.3239937
8: 7.8777013, 11.0664644, 7.8561664, 11.0790901, -3.0313361, 3.1169145
9: -7.1286869, -3.2516174, -7.1246834, -3.2466125, -3.6624155, 3.5207400

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B1_A1_B1_A1

### Relational analysis result of IS_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1339271, upper bound: 2.1046320
time: 9.11 seconds

## Relational analysis of IS_B2_B1_A1_B1_A2

### Relational analysis result of IS_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1339272, upper bound: 2.1068658
time: 8.84 seconds

## BFS IS instance: IS_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1839566, -1.7835217, -6.1962519, -1.7841630, -4.3808451, 4.3918371
1: -15.2628155, -10.7436018, -15.2972393, -10.7122097, -4.4057312, 4.5174923
2: -9.0587969, -4.6700068, -9.1495333, -4.5761361, -4.1734180, 4.1678128
3: -7.5691686, -3.6043870, -7.5837040, -3.5838339, -3.9853346, 3.9793169
4: -12.2045078, -7.3999367, -12.2653751, -7.3659964, -4.8385115, 4.8654385
5: -5.9820123, -2.2355981, -6.0167756, -2.2000248, -3.6590719, 3.6667771
6: -13.7659960, -9.3151302, -13.7735767, -9.2978630, -3.9295177, 3.9004207
7: -10.2170382, -5.8796635, -10.2340193, -5.8906541, -4.3263841, 4.3543558
8: 7.8695936, 11.0690136, 7.8268309, 11.1009655, -3.0579927, 3.1388865
9: -7.1345873, -3.2473459, -7.1479235, -3.2283878, -3.6867986, 3.5635738

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B1_A1_B2_A1

### Relational analysis result of IS_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354860, upper bound: 2.1164088
time: 5.63 seconds

## Relational analysis of IS_B2_B1_A1_B2_A2

### Relational analysis result of IS_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354861, upper bound: 2.1186412
time: 5.57 seconds

## BFS IS instance: IS_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1924620, -1.7616458, -6.1362791, -1.8214746, -4.3709874, 4.3746333
1: -15.2947197, -10.7078934, -15.2382622, -10.7401867, -4.5279388, 4.5202270
2: -9.1565723, -4.5728912, -9.1377182, -4.6134310, -4.2200460, 4.2407975
3: -7.6119432, -3.5602407, -7.5652986, -3.5954003, -3.9926777, 3.9824629
4: -12.2729511, -7.3625507, -12.2253914, -7.3918042, -4.8811469, 4.8628407
5: -6.0269818, -2.1939924, -5.9956317, -2.2169721, -3.6747313, 3.6606464
6: -13.7975388, -9.2646294, -13.7303085, -9.3233337, -3.9080734, 3.8974562
7: -10.2584991, -5.8715467, -10.2096767, -5.9124813, -4.3460178, 4.3381300
8: 7.8208055, 11.1100655, 7.8561664, 11.0790901, -3.1571736, 3.1523070
9: -7.1655450, -3.2284203, -7.1246834, -3.2466125, -3.7216120, 3.6913252

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1147956, upper bound: 2.1237684
time: 5.89 seconds

## Relational analysis of IS_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1147955, upper bound: 2.1259575
time: 5.75 seconds

## BFS IS instance: IS_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.2093658, -1.7585182, -6.1962519, -1.7841630, -4.4252028, 4.4377337
1: -15.3110104, -10.7050266, -15.2972393, -10.7122097, -4.5727348, 4.5822020
2: -9.1579075, -4.5629139, -9.1495333, -4.5761361, -4.2558708, 4.2656879
3: -7.6149983, -3.5575323, -7.5837040, -3.5838339, -4.0071583, 4.0023646
4: -12.2786884, -7.3551311, -12.2653751, -7.3659964, -4.9126921, 4.9102440
5: -6.0307689, -2.1897502, -6.0167756, -2.2000248, -3.6982794, 3.7085919
6: -13.8098650, -9.2619419, -13.7735767, -9.2978630, -3.9463539, 3.9428387
7: -10.2620068, -5.8655272, -10.2340193, -5.8906541, -4.3713527, 4.3684921
8: 7.8126569, 11.1126156, 7.8268309, 11.1009655, -3.1839399, 3.1749573
9: -7.1714444, -3.2241068, -7.1479235, -3.2283878, -3.7460213, 3.7343035

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B1_A2_B2_A1

### Relational analysis result of IS_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1354826
time: 9.56 seconds

## Relational analysis of IS_B2_B1_A2_B2_A2

### Relational analysis result of IS_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1376346
time: 9.65 seconds

## BFS IS instance: IS_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1710329, -1.7848697, -6.1533027, -1.7940540, -4.3625755, 4.3553324
1: -15.2476873, -10.7453880, -15.2530632, -10.7319393, -4.3742781, 4.4749737
2: -9.0581474, -4.6773758, -9.1467915, -4.5976763, -4.1530409, 4.1501269
3: -7.5674276, -3.5967989, -7.5978894, -3.5588515, -3.9854574, 4.0010905
4: -12.2017918, -7.4064302, -12.2417507, -7.3800268, -4.8217649, 4.8353205
5: -5.9800744, -2.2364011, -6.0114608, -2.2032707, -3.6513538, 3.6455755
6: -13.7581854, -9.3039341, -13.7709198, -9.2735939, -3.9092579, 3.8989558
7: -10.2225561, -5.8842278, -10.2466278, -5.8859382, -4.3366179, 4.3624001
8: 7.8764839, 11.0702953, 7.8407817, 11.0946217, -3.0435085, 3.1273916
9: -7.1326485, -3.2509148, -7.1516843, -3.2416608, -3.6731858, 3.5519452

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B2_A1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1339228, upper bound: 2.1046319
time: 11.58 seconds

## Relational analysis of IS_B2_B2_A1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1339227, upper bound: 2.1068662
time: 5.53 seconds

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1879392, -1.7817011, -6.2133484, -1.7567105, -4.4139919, 4.4138384
1: -15.2639389, -10.7425194, -15.3121214, -10.7039604, -4.4208622, 4.5375738
2: -9.0594826, -4.6674361, -9.1586027, -4.5603623, -4.1889868, 4.1750398
3: -7.5704589, -3.5940895, -7.6162753, -3.5472648, -3.9999113, 4.0221858
4: -12.2075672, -7.3990169, -12.2817097, -7.3542099, -4.8533573, 4.8826928
5: -5.9838691, -2.2321620, -6.0325961, -2.1863008, -3.6748457, 3.6851659
6: -13.7705021, -9.3012848, -13.8142519, -9.2481308, -3.9475279, 3.9430194
7: -10.2260056, -5.8782077, -10.2709389, -5.8641076, -4.3618979, 4.3927312
8: 7.8683767, 11.0728436, 7.8114467, 11.1164694, -3.0701568, 3.1493571
9: -7.1385517, -3.2466426, -7.1753292, -3.2234340, -3.6975842, 3.5949292

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B2_A1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354817, upper bound: 2.1164092
time: 5.88 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.1354828, upper bound: 2.1186411
time: 5.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.30 seconds
IS_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1147988
IS_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1147976
IS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164090, upper bound: 2.1164157
IS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164089, upper bound: 2.1164139
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1170273
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046328, upper bound: 2.1170258
IS_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164089, upper bound: 2.1186438
IS_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164090, upper bound: 2.1186418
IS_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046324, upper bound: 2.1339278
IS_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046324, upper bound: 2.1339263
IS_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164085, upper bound: 2.1354880
IS_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164086, upper bound: 2.1354857
IS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046324, upper bound: 2.1361181
IS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1046325, upper bound: 2.1361169
IS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164086, upper bound: 2.1376352
IS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164086, upper bound: 2.1376346
IS_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1339271, upper bound: 2.1046320
IS_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1339272, upper bound: 2.1068658
IS_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1354860, upper bound: 2.1164088
IS_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1354861, upper bound: 2.1186412
IS_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1147956, upper bound: 2.1237684
IS_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1147955, upper bound: 2.1259575
IS_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1354826
IS_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1164110, upper bound: 2.1376346
IS_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1339228, upper bound: 2.1046319
IS_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1339227, upper bound: 2.1068662
IS_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1354817, upper bound: 2.1164092
IS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.30
Output dim: 8, lower bound: -2.1354828, upper bound: 2.1186411
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.30
Output dim: 8, lower bound: -2.1164220, upper bound: 2.1354980
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.30
Output dim: 8, lower bound: -2.1164220, upper bound: 2.1376512
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=3.226428508758545
rel_dist={8: [-2.137668074409355, 2.1376683881402094]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6195
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 846
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -2.0089420, upper bound: 1.9922138
time: 5.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090751, upper bound: 2.0090750
time: 10.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.23 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 16.23
Output dim: 8, lower bound: -2.0089420, upper bound: 1.9922138
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.23
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

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090718, upper bound: 2.0070873
time: 5.56 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -2.0090718, upper bound: 2.0090708
time: 6.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.96 seconds
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.96
Output dim: 8, lower bound: -2.0090718, upper bound: 2.0070873
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.96
Output dim: 8, lower bound: -2.0090718, upper bound: 2.0090708

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -6.1962962, -1.7841535, -6.2086058, -1.7588615, -4.3293209, 4.3513985
1: -15.2972927, -10.7122021, -15.3108034, -10.7052288, -4.4468651, 4.4613204
2: -9.1495380, -4.5761046, -9.1578407, -4.5634089, -4.1856136, 4.1813526
3: -7.5837145, -3.5838249, -7.6147676, -3.5595069, -3.8994980, 3.9437895
4: -12.2653923, -7.3659816, -12.2781172, -7.3552775, -4.9101148, 4.9121356
5: -6.0167847, -2.2000177, -6.0304213, -2.1903858, -3.6110868, 3.6100173
6: -13.7736197, -9.2978554, -13.8090162, -9.2645769, -3.8481212, 3.8260899
7: -10.2340288, -5.8906298, -10.2602978, -5.8658032, -4.3682256, 4.3696680
8: 7.8268032, 11.1009731, 7.8128877, 11.1118860, -3.1108184, 3.1036983
9: -7.1479387, -3.2283797, -7.1706953, -3.2242284, -3.6261864, 3.6566544

Time for backsubstitution: 13.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9922091, upper bound: 2.0069511
time: 5.76 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9922092, upper bound: 2.0070872
time: 9.16 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -6.2133961, -1.7566996, -6.2134061, -1.7566948, -4.3512039, 4.3847675
1: -15.3121748, -10.7039547, -15.3121910, -10.7039490, -4.4666119, 4.4767880
2: -9.1586056, -4.5603304, -9.1586752, -4.5603209, -4.2015109, 4.1962705
3: -7.6162853, -3.5472577, -7.6163087, -3.5472422, -3.9468994, 3.9506721
4: -12.2817230, -7.3541961, -12.2817430, -7.3541613, -4.9275618, 4.9275470
5: -6.0326071, -2.1862943, -6.0326138, -2.1862650, -3.6338177, 3.6260204
6: -13.8142920, -9.2481222, -13.8143024, -9.2480869, -3.8933344, 3.8427863
7: -10.2709503, -5.8640852, -10.2709684, -5.8640809, -4.4068694, 4.4068832
8: 7.8114200, 11.1164789, 7.8114090, 11.1164932, -3.1216769, 3.1253490
9: -7.1753473, -3.2234244, -7.1753588, -3.2234159, -3.6619816, 3.6679173

Time for backsubstitution: 13.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6195
type: A, layer: 1, pos: 4555
type: B, layer: 1, pos: 4555
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5761
type: A, layer: 1, pos: 5761
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 846
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 847
type: B, layer: 1, pos: 847
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 846

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6195

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9922091, upper bound: 2.0089388
time: 5.86 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.9922091, upper bound: 2.0089398
time: 6.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.36 seconds
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 25.36
Output dim: 8, lower bound: -1.9922091, upper bound: 2.0069511
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 25.36
Output dim: 8, lower bound: -1.9922092, upper bound: 2.0070872
IS_A2_A2_B1, status: Status.VERIFIED, split count: 3, time: 25.36
Output dim: 8, lower bound: -1.9922091, upper bound: 2.0089388
IS_A2_A2_B2, status: Status.VERIFIED, split count: 3, time: 25.36
Output dim: 8, lower bound: -1.9922091, upper bound: 2.0089398
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=3.14231538772583
rel_dist={8: [-2.00908813677729, 2.009087543266727]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1759.18 seconds
