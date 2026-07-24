## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.872541919
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.3938293, 2.3938293)
1: (-17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.6634283, 3.6634278)
2: (-3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.7616858, 2.7616858)
3: (-10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9296875, 2.9296875)
4: (-12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.5041504, 3.5041494)
5: (-4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.3018072, 2.3018072)
6: (-3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.5280423, 2.5280423)
7: (-9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.9013386, 3.9013381)
8: (-2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2600141, 2.2600141)
9: (-4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.7319930, 2.7319930)

## BASE Result
execution time: IAR + LP analysis = 15.48 + 34.43 = 49.91 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3550.09 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.027801513671875
rel_dist={0: [-1.1944731507389008, 1.1944727062795257]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8442518711090088
rel_dist={0: [-0.8742898229747036, 0.874290832632278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.7218856811523438
rel_dist={0: [-0.6370208004750442, 0.6370207494992766]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7830688953399658
rel_dist={0: [-0.7572591028880957, 0.7572587984536243]}

## Binary Search Result
Binary search time: 222.65 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3327.43 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2852455
time: 5.40 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2945697
time: 5.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2852455
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2945697

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.6417065, 9.0272636, -2.0435870, 2.0669985
1: -17.4738808, -13.8005133, -17.4824867, -13.7725754, -3.0394354, 3.1404214
2: -3.2627931, -0.5239117, -3.2724638, -0.5164485, -2.6353097, 2.6314273
3: -10.8545074, -7.9736691, -10.8632708, -7.9463143, -2.7451506, 2.8622994
4: -12.5303125, -9.0268316, -12.5367823, -9.0183058, -3.0763030, 3.0654564
5: -4.9079676, -2.6729443, -4.9523115, -2.6663408, -2.1727982, 2.1036384
6: -3.0038805, -0.5669265, -3.0646939, -0.5582647, -2.3757339, 2.3318439
7: -9.3258791, -5.4602871, -9.3383026, -5.4104347, -3.3345299, 3.4343224
8: -2.5947652, -0.3486171, -2.6001096, -0.3434892, -2.2198467, 2.2274809
9: -4.4674053, -1.7695539, -4.4771047, -1.7534380, -2.4826975, 2.4709775

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852456
time: 5.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852459
time: 5.59 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.6375146, 9.0313377, -2.1341560, 2.0917320
1: -17.5395622, -13.7612896, -17.4851475, -13.7643375, -3.2358541, 3.1837478
2: -3.2866726, -0.4902350, -3.2759018, -0.5142194, -2.6566839, 2.6703506
3: -10.9079905, -7.9272232, -10.8677940, -7.9381132, -2.9497442, 2.9120026
4: -12.5829248, -8.9888010, -12.5387926, -9.0154028, -3.1360264, 3.1351595
5: -4.9724512, -2.5807762, -4.9653778, -2.6635828, -2.2409835, 2.2771697
6: -3.0983086, -0.4341698, -3.0826294, -0.5545921, -2.4685612, 2.5021944
7: -9.4304581, -5.3878284, -9.3434544, -5.3957100, -3.5679603, 3.5118098
8: -2.6340675, -0.3340945, -2.6018820, -0.3418698, -2.2663522, 2.2445097
9: -4.5135813, -1.7374017, -4.4801087, -1.7481202, -2.5343719, 2.5265160

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852482, upper bound: 1.2945693
time: 6.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2945696
time: 5.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.36 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852456
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852459
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 0, lower bound: -1.2852482, upper bound: 1.2945693
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2945696

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.6551013, 9.0135641, -2.0295057, 2.0295062
1: -17.4738808, -13.8005133, -17.4738808, -13.8005133, -3.0104165, 3.0104156
2: -3.2627931, -0.5239117, -3.2627931, -0.5239117, -2.6267295, 2.6267295
3: -10.8545074, -7.9736691, -10.8545074, -7.9736691, -2.7167091, 2.7167087
4: -12.5303125, -9.0268316, -12.5303125, -9.0268316, -3.0576868, 3.0576878
5: -4.9079676, -2.6729443, -4.9079676, -2.6729443, -2.0589428, 2.0589426
6: -3.0038805, -0.5669265, -3.0038805, -0.5669265, -2.2656853, 2.2656856
7: -9.3258791, -5.4602871, -9.3258791, -5.4602871, -3.2865877, 3.2865877
8: -2.5947652, -0.3486171, -2.5947652, -0.3486171, -2.2145286, 2.2145286
9: -4.4674053, -1.7695539, -4.4674053, -1.7695539, -2.4597116, 2.4597118

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852388
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852423, upper bound: 1.2852389
time: 5.50 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.5904884, 9.0354805, -2.0519533, 2.1159120
1: -17.4738808, -13.8005133, -17.5395622, -13.7612896, -3.0524530, 3.1997461
2: -3.2627931, -0.5239117, -3.2866726, -0.4902350, -2.6616263, 2.6447611
3: -10.8545074, -7.9736691, -10.9079905, -7.9272232, -2.7652917, 2.9099016
4: -12.5303125, -9.0268316, -12.5829248, -8.9888010, -3.0942736, 3.1125288
5: -4.9079676, -2.6729443, -4.9724512, -2.5807762, -2.2209535, 2.1273835
6: -3.0038805, -0.5669265, -3.0983086, -0.4341698, -2.4225490, 2.3617415
7: -9.3258791, -5.4602871, -9.4304581, -5.3878284, -3.3632398, 3.5048056
8: -2.5947652, -0.3486171, -2.6340675, -0.3340945, -2.2306147, 2.2581644
9: -4.4674053, -1.7695539, -4.5135813, -1.7374017, -2.4955077, 2.5065489

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852426, upper bound: 1.2809054
time: 5.46 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852425, upper bound: 1.2852392
time: 5.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.6551013, 9.0135641, -2.1159122, 2.0519536
1: -17.5395622, -13.7612896, -17.4738808, -13.8005133, -3.1997457, 3.0524526
2: -3.2866726, -0.4902350, -3.2627931, -0.5239117, -2.6447611, 2.6616263
3: -10.9079905, -7.9272232, -10.8545074, -7.9736691, -2.9099021, 2.7652917
4: -12.5829248, -8.9888010, -12.5303125, -9.0268316, -3.1125283, 3.0942736
5: -4.9724512, -2.5807762, -4.9079676, -2.6729443, -2.1273832, 2.2209537
6: -3.0983086, -0.4341698, -3.0038805, -0.5669265, -2.3617413, 2.4225488
7: -9.4304581, -5.3878284, -9.3258791, -5.4602871, -3.5048056, 3.3632402
8: -2.6340675, -0.3340945, -2.5947652, -0.3486171, -2.2581644, 2.2306147
9: -4.5135813, -1.7374017, -4.4674053, -1.7695539, -2.5065484, 2.4955077

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639
time: 7.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852390, upper bound: 1.2945637
time: 5.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.5904884, 9.0354805, -2.1356874, 2.1356871
1: -17.5395622, -13.7612896, -17.5395622, -13.7612896, -3.2371330, 3.2371321
2: -3.2866726, -0.4902350, -3.2866726, -0.4902350, -2.6731591, 2.6731591
3: -10.9079905, -7.9272232, -10.9079905, -7.9272232, -2.9600568, 2.9600563
4: -12.5829248, -8.9888010, -12.5829248, -8.9888010, -3.1557336, 3.1557341
5: -4.9724512, -2.5807762, -4.9724512, -2.5807762, -2.2871881, 2.2871883
6: -3.0983086, -0.4341698, -3.0983086, -0.4341698, -2.5120704, 2.5120707
7: -9.4304581, -5.3878284, -9.4304581, -5.3878284, -3.5786462, 3.5786459
8: -2.6340675, -0.3340945, -2.6340675, -0.3340945, -2.2735186, 2.2735186
9: -4.5135813, -1.7374017, -4.5135813, -1.7374017, -2.5374427, 2.5374427

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2902592
time: 5.61 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2945645
time: 5.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.10 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852388
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2852423, upper bound: 1.2852389
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2852426, upper bound: 1.2809054
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2852425, upper bound: 1.2852392
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2852390, upper bound: 1.2945637
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2902592
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2945645

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.6551013, 9.0135641, -2.0229845, 2.0285563
1: -17.4648304, -13.8021736, -17.4738808, -13.8005133, -3.0012460, 3.0088005
2: -3.2573876, -0.5272779, -3.2627931, -0.5239117, -2.6213293, 2.6236081
3: -10.8511620, -7.9760609, -10.8545074, -7.9736691, -2.7129545, 2.7146583
4: -12.5276699, -9.0527420, -12.5303125, -9.0268316, -3.0550203, 3.0316749
5: -4.9013195, -2.6738970, -4.9079676, -2.6729443, -2.0513868, 2.0579462
6: -2.9942713, -0.5683670, -3.0038805, -0.5669265, -2.2559278, 2.2643287
7: -9.3230848, -5.4961648, -9.3258791, -5.4602871, -3.2838221, 3.2505813
8: -2.5841184, -0.3503575, -2.5947652, -0.3486171, -2.2037711, 2.2125154
9: -4.4652653, -1.7884412, -4.4674053, -1.7695539, -2.4576168, 2.4408369

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809105, upper bound: 1.2809084
time: 9.05 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852422
time: 9.67 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.6551037, 9.0135641, -2.0317268, 2.0435870
1: -17.4762993, -13.7852192, -17.4738770, -13.8005142, -3.0132475, 3.0259347
2: -3.2660034, -0.5133837, -3.2627921, -0.5239124, -2.6303821, 2.6381133
3: -10.8567801, -7.9611416, -10.8545065, -7.9736710, -2.7185025, 2.7320914
4: -12.6000013, -9.0257072, -12.5303135, -9.0268450, -3.1281180, 3.0589800
5: -4.9100409, -2.6617544, -4.9079633, -2.6729436, -2.0611830, 2.0704288
6: -3.0051947, -0.5469360, -3.0038781, -0.5669279, -2.2671719, 2.2864118
7: -9.4240265, -5.4593987, -9.3258781, -5.4603081, -3.3481007, 3.2888293
8: -2.5963993, -0.3227391, -2.5947604, -0.3486171, -2.2162952, 2.2257514
9: -4.5145378, -1.7673830, -4.4674039, -1.7695656, -2.4907446, 2.4622960

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852447, upper bound: 1.2809081
time: 5.83 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852425, upper bound: 1.2852420
time: 5.25 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.5967369, 9.0346031, -2.0510015, 2.1093998
1: -17.4738808, -13.8005133, -17.5305214, -13.7629414, -3.0508370, 3.1905742
2: -3.2627931, -0.5239117, -3.2812915, -0.4936304, -2.6584811, 2.6393518
3: -10.8545074, -7.9736691, -10.9046412, -7.9296236, -2.7632427, 2.9061236
4: -12.5303125, -9.0268316, -12.5802679, -9.0147047, -3.0682664, 3.1098042
5: -4.9079676, -2.6729443, -4.9658098, -2.5817404, -2.2198782, 2.1198196
6: -3.0038805, -0.5669265, -3.0887170, -0.4356251, -2.4211736, 2.3519998
7: -9.3258791, -5.4602871, -9.4276257, -5.4237018, -3.3272424, 3.5020263
8: -2.5947652, -0.3486171, -2.6234269, -0.3358107, -2.2286139, 2.2474096
9: -4.4674053, -1.7695539, -4.5114365, -1.7562879, -2.4766326, 2.5044360

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902604, upper bound: 1.2809049
time: 7.51 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902586, upper bound: 1.2809055
time: 5.69 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 6.6551037, 9.0135641, 6.5878325, 9.0499220, -2.0660729, 2.1181343
1: -17.4738770, -13.8005142, -17.5419140, -13.7459955, -3.0679674, 3.2025061
2: -3.2627921, -0.5239124, -3.2899241, -0.4797205, -2.6686137, 2.6484423
3: -10.8545065, -7.9736710, -10.9102325, -7.9146242, -2.7807436, 2.9116020
4: -12.5303135, -9.0268450, -12.6525669, -8.9876604, -3.0955801, 3.1577501
5: -4.9079633, -2.6729436, -4.9745350, -2.5696204, -2.2230577, 2.1296387
6: -3.0038781, -0.5669279, -3.0996435, -0.4142132, -2.4271469, 2.3632512
7: -9.3258781, -5.4603081, -9.5286961, -5.3869095, -3.3655005, 3.5181694
8: -2.5947604, -0.3486171, -2.6356602, -0.3081894, -2.2403173, 2.2599356
9: -4.4674039, -1.7695656, -4.5606918, -1.7352251, -2.4980955, 2.5151279

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2944349, upper bound: 1.2782377
time: 6.20 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945561, upper bound: 1.2852311
time: 6.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.6551013, 9.0135641, -2.1094000, 2.0510011
1: -17.5305214, -13.7629414, -17.4738808, -13.8005133, -3.1905732, 3.0508366
2: -3.2812915, -0.4936304, -3.2627931, -0.5239117, -2.6393518, 2.6584811
3: -10.9046412, -7.9296236, -10.8545074, -7.9736691, -2.9061236, 2.7632418
4: -12.5802679, -9.0147047, -12.5303125, -9.0268316, -3.1098046, 3.0682659
5: -4.9658098, -2.5817404, -4.9079676, -2.6729443, -2.1198196, 2.2198784
6: -3.0887170, -0.4356251, -3.0038805, -0.5669265, -2.3520000, 2.4211736
7: -9.4276257, -5.4237018, -9.3258791, -5.4602871, -3.5020266, 3.3272419
8: -2.6234269, -0.3358107, -2.5947652, -0.3486171, -2.2474098, 2.2286134
9: -4.5114365, -1.7562879, -4.4674053, -1.7695539, -2.5044360, 2.4766328

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902582
time: 7.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945642
time: 8.56 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 6.5878325, 9.0499220, 6.6551037, 9.0135641, -2.1181343, 2.0660729
1: -17.5419140, -13.7459955, -17.4738770, -13.8005142, -3.2025061, 3.0679674
2: -3.2899241, -0.4797205, -3.2627921, -0.5239124, -2.6484423, 2.6686134
3: -10.9102325, -7.9146242, -10.8545065, -7.9736710, -2.9116011, 2.7807426
4: -12.6525669, -8.9876604, -12.5303135, -9.0268450, -3.1577501, 3.0955806
5: -4.9745350, -2.5696204, -4.9079633, -2.6729436, -2.1296387, 2.2230580
6: -3.0996435, -0.4142132, -3.0038781, -0.5669279, -2.3632507, 2.4271467
7: -9.5286961, -5.3869095, -9.3258781, -5.4603081, -3.5181694, 3.3655005
8: -2.6356602, -0.3081894, -2.5947604, -0.3486171, -2.2599354, 2.2403173
9: -4.5606918, -1.7352251, -4.4674039, -1.7695656, -2.5151277, 2.4980960

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2782377, upper bound: 1.2944349
time: 8.31 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852309, upper bound: 1.2945556
time: 5.18 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.5967369, 9.0346031, -2.1347339, 2.1291573
1: -17.5395622, -13.7612896, -17.5305214, -13.7629414, -3.2355175, 3.2279596
2: -3.2866726, -0.4902350, -3.2812915, -0.4936304, -2.6700134, 2.6677048
3: -10.9079905, -7.9272232, -10.9046412, -7.9296236, -2.9580698, 2.9562769
4: -12.5829248, -8.9888010, -12.5802679, -9.0147047, -3.1297121, 3.1530094
5: -4.9724512, -2.5807762, -4.9658098, -2.5817404, -2.2861128, 2.2797048
6: -3.0983086, -0.4341698, -3.0887170, -0.4356251, -2.5106959, 2.5024893
7: -9.4304581, -5.3878284, -9.4276257, -5.4237018, -3.5426450, 3.5758667
8: -2.6340675, -0.3340945, -2.6234269, -0.3358107, -2.2714972, 2.2627637
9: -4.5135813, -1.7374017, -4.5114365, -1.7562879, -2.5185690, 2.5353990

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902609
time: 8.02 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902585
time: 5.34 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 6.5904913, 9.0354805, 6.5878325, 9.0499220, -2.1399727, 2.1378248
1: -17.5395584, -13.7612877, -17.5419140, -13.7459955, -3.2526455, 3.2398930
2: -3.2866712, -0.4902354, -3.2899241, -0.4797205, -2.6845374, 2.6767437
3: -10.9079885, -7.9272246, -10.9102325, -7.9146242, -2.9758253, 2.9617562
4: -12.5829220, -8.9888144, -12.6525669, -8.9876604, -3.1570168, 3.2040372
5: -4.9724493, -2.5807762, -4.9745350, -2.5696204, -2.2892928, 2.2892625
6: -3.0983055, -0.4341702, -3.0996435, -0.4142132, -2.5166683, 2.5136416
7: -9.4304562, -5.3878465, -9.5286961, -5.3869095, -3.5809965, 3.5920107
8: -2.6340632, -0.3340960, -2.6356602, -0.3081894, -2.2763777, 2.2752888
9: -4.5135813, -1.7374126, -4.5606918, -1.7352251, -2.5399842, 2.5525658

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945640
time: 8.20 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639
time: 5.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.70 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809105, upper bound: 1.2809084
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852422
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2852447, upper bound: 1.2809081
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2852425, upper bound: 1.2852420
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2902604, upper bound: 1.2809049
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2902586, upper bound: 1.2809055
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2944349, upper bound: 1.2782377
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2945561, upper bound: 1.2852311
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902582
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945642
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2782377, upper bound: 1.2944349
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2852309, upper bound: 1.2945556
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902609
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902585
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945640
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.6613379, 9.0126896, -2.0220356, 2.0220351
1: -17.4648304, -13.8021736, -17.4648304, -13.8021736, -2.9996314, 2.9996309
2: -3.2573876, -0.5272779, -3.2573876, -0.5272779, -2.6182079, 2.6182077
3: -10.8511620, -7.9760609, -10.8511620, -7.9760609, -2.7109051, 2.7109046
4: -12.5276699, -9.0527420, -12.5276699, -9.0527420, -3.0290089, 3.0290079
5: -4.9013195, -2.6738970, -4.9013195, -2.6738970, -2.0503907, 2.0503907
6: -2.9942713, -0.5683670, -2.9942713, -0.5683670, -2.2545710, 2.2545707
7: -9.3230848, -5.4961648, -9.3230848, -5.4961648, -3.2478142, 3.2478149
8: -2.5841184, -0.3503575, -2.5841184, -0.3503575, -2.2017579, 2.2017579
9: -4.4652653, -1.7884412, -4.4652653, -1.7884412, -2.4387417, 2.4387417

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2739079, upper bound: 1.2807807
time: 5.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809002, upper bound: 1.2809055
time: 6.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.6523466, 9.0279541, -2.0370667, 2.0307484
1: -17.4648304, -13.8021736, -17.4762993, -13.7852192, -3.0167675, 3.0115728
2: -3.2573876, -0.5272779, -3.2660034, -0.5133837, -2.6327152, 2.6272507
3: -10.8511620, -7.9760609, -10.8567801, -7.9611416, -2.7283392, 2.7164435
4: -12.5276699, -9.0527420, -12.6000013, -9.0257072, -3.0560741, 3.1021109
5: -4.9013195, -2.6738970, -4.9100409, -2.6617544, -2.0628748, 2.0601535
6: -2.9942713, -0.5683670, -3.0051947, -0.5469360, -2.2766552, 2.2657673
7: -9.3230848, -5.4961648, -9.4240265, -5.4593987, -3.2857652, 3.3120816
8: -2.5841184, -0.3503575, -2.5963993, -0.3227391, -2.2149920, 2.2141767
9: -4.4652653, -1.7884412, -4.5145378, -1.7673830, -2.4600320, 2.4718573

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2739079, upper bound: 1.2851094
time: 5.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809002, upper bound: 1.2852340
time: 6.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.6613379, 9.0126896, -2.0307484, 2.0370667
1: -17.4762993, -13.7852192, -17.4648304, -13.8021736, -3.0115733, 3.0167675
2: -3.2660034, -0.5133837, -3.2573876, -0.5272779, -2.6272507, 2.6327152
3: -10.8567801, -7.9611416, -10.8511620, -7.9760609, -2.7164440, 2.7283392
4: -12.6000013, -9.0257072, -12.5276699, -9.0527420, -3.1021099, 3.0560741
5: -4.9100409, -2.6617544, -4.9013195, -2.6738970, -2.0601535, 2.0628748
6: -3.0051947, -0.5469360, -2.9942713, -0.5683670, -2.2657671, 2.2766554
7: -9.4240265, -5.4593987, -9.3230848, -5.4961648, -3.3120813, 3.2857652
8: -2.5963993, -0.3227391, -2.5841184, -0.3503575, -2.2141767, 2.2149916
9: -4.5145378, -1.7673830, -4.4652653, -1.7884412, -2.4718575, 2.4600322

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2851091, upper bound: 1.2739100
time: 10.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852342, upper bound: 1.2808998
time: 5.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.6523466, 9.0279541, -2.0378289, 2.0378287
1: -17.4762993, -13.7852192, -17.4762993, -13.7852192, -3.0150857, 3.0150862
2: -3.2660034, -0.5133837, -3.2660034, -0.5133837, -2.6391268, 2.6391268
3: -10.8567801, -7.9611416, -10.8567801, -7.9611416, -2.7315264, 2.7315269
4: -12.6000013, -9.0257072, -12.6000013, -9.0257072, -3.0739484, 3.0739479
5: -4.9100409, -2.6617544, -4.9100409, -2.6617544, -2.0643153, 2.0643153
6: -3.0051947, -0.5469360, -3.0051947, -0.5469360, -2.2768538, 2.2768536
7: -9.4240265, -5.4593987, -9.4240265, -5.4593987, -3.3172102, 3.3172112
8: -2.5963993, -0.3227391, -2.5963993, -0.3227391, -2.2193131, 2.2193131
9: -4.5145378, -1.7673830, -4.5145378, -1.7673830, -2.4745955, 2.4745955

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2782411, upper bound: 1.2808100
time: 7.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852344, upper bound: 1.2809332
time: 5.86 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.5967369, 9.0346031, -2.0444798, 2.1084304
1: -17.4648304, -13.8021736, -17.5305214, -13.7629414, -3.0416675, 3.1889596
2: -3.2573876, -0.5272779, -3.2812915, -0.4936304, -2.6530809, 2.6362300
3: -10.8511620, -7.9760609, -10.9046412, -7.9296236, -2.7594881, 2.9041381
4: -12.5276699, -9.0527420, -12.5802679, -9.0147047, -3.0655980, 3.0837922
5: -4.9013195, -2.6738970, -4.9658098, -2.5817404, -2.2124095, 2.1188228
6: -2.9942713, -0.5683670, -3.0887170, -0.4356251, -2.4115887, 2.3506432
7: -9.3230848, -5.4961648, -9.4276257, -5.4237018, -3.3244758, 3.4660184
8: -2.5841184, -0.3503575, -2.6234269, -0.3358107, -2.2178559, 2.2453783
9: -4.4652653, -1.7884412, -4.5114365, -1.7562879, -2.4745669, 2.4855614

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2901329, upper bound: 1.2739043
time: 6.15 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902561, upper bound: 1.2808966
time: 9.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.5967369, 9.0346031, -2.0531931, 2.1123831
1: -17.4762993, -13.7852192, -17.5305214, -13.7629414, -3.0536094, 3.2060947
2: -3.2660034, -0.5133837, -3.2812915, -0.4936304, -2.6621237, 2.6507845
3: -10.8567801, -7.9611416, -10.9046412, -7.9296236, -2.7650270, 2.9218354
4: -12.6000013, -9.0257072, -12.5802679, -9.0147047, -3.1387029, 3.1108584
5: -4.9100409, -2.6617544, -4.9658098, -2.5817404, -2.2218971, 2.1292305
6: -3.0051947, -0.5469360, -3.0887170, -0.4356251, -2.4226718, 2.3630300
7: -9.4240265, -5.4593987, -9.4276257, -5.4237018, -3.3705111, 3.5040262
8: -2.5963993, -0.3227391, -2.6234269, -0.3358107, -2.2302747, 2.2502406
9: -4.5145378, -1.7673830, -4.5114365, -1.7562879, -2.4991338, 2.5068519

Time for backsubstitution: 12.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2901329, upper bound: 1.2739052
time: 5.51 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902561, upper bound: 1.2808973
time: 5.68 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 6.6599426, 9.0120716, 6.6190963, 9.0374622, -2.0430336, 2.0835156
1: -17.4670715, -13.8027258, -17.4985886, -13.7623196, -3.0344477, 3.1571932
2: -3.2601101, -0.5296283, -3.2717190, -0.5150280, -2.6285408, 2.6218066
3: -10.8510284, -7.9778361, -10.8885679, -7.9413786, -2.7420778, 2.8776736
4: -12.5209999, -9.0285950, -12.5936260, -9.0024395, -3.0733829, 3.0929227
5: -4.9052739, -2.6745539, -4.9584179, -2.5823247, -2.2005630, 2.1127305
6: -2.9961185, -0.5684438, -3.0514343, -0.4269853, -2.3855939, 2.3095269
7: -9.3222227, -5.4665804, -9.5002108, -5.4257102, -3.3214622, 3.4663906
8: -2.5869083, -0.3497086, -2.5898132, -0.3175340, -2.2125077, 2.2185102
9: -4.4593191, -1.7708025, -4.5108137, -1.7477043, -2.4765506, 2.4591517

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 5773

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2929220, upper bound: 1.2740528
time: 6.72 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2901273, upper bound: 1.2782373
time: 5.44 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2901292, upper bound: 1.2739378
time: 7.02 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 6.6551037, 9.0135641, 6.5878363, 9.0499210, -2.0660715, 2.1136343
1: -17.4738770, -13.8005142, -17.5419102, -13.7459946, -3.0657859, 3.2067356
2: -3.2627921, -0.5239124, -3.2899222, -0.4797257, -2.6596479, 2.6477816
3: -10.8545065, -7.9736710, -10.9102306, -7.9146261, -2.7807388, 2.9253969
4: -12.5303135, -9.0268450, -12.6525583, -8.9876614, -3.0955801, 3.1529245
5: -4.9079633, -2.6729436, -4.9745345, -2.5696211, -2.2219329, 2.1351514
6: -3.0038781, -0.5669279, -3.0996401, -0.4142132, -2.4253969, 2.3534222
7: -9.3258781, -5.4603081, -9.5286932, -5.3869123, -3.3645430, 3.5155551
8: -2.5947604, -0.3486171, -2.6356559, -0.3081908, -2.2382193, 2.2462263
9: -4.4674039, -1.7695656, -4.5606894, -1.7352264, -2.4980946, 2.5015621

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 5773

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945550, upper bound: 1.2847087
time: 7.02 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945550, upper bound: 1.2852301
time: 6.27 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.6613379, 9.0126896, -2.1084299, 2.0444798
1: -17.5305214, -13.7629414, -17.4648304, -13.8021736, -3.1889596, 3.0416665
2: -3.2812915, -0.4936304, -3.2573876, -0.5272779, -2.6362300, 2.6530812
3: -10.9046412, -7.9296236, -10.8511620, -7.9760609, -2.9041386, 2.7594881
4: -12.5802679, -9.0147047, -12.5276699, -9.0527420, -3.0837922, 3.0655990
5: -4.9658098, -2.5817404, -4.9013195, -2.6738970, -2.1188231, 2.2124100
6: -3.0887170, -0.4356251, -2.9942713, -0.5683670, -2.3506432, 2.4115891
7: -9.4276257, -5.4237018, -9.3230848, -5.4961648, -3.4660177, 3.3244758
8: -2.6234269, -0.3358107, -2.5841184, -0.3503575, -2.2453780, 2.2178559
9: -4.5114365, -1.7562879, -4.4652653, -1.7884412, -2.4855614, 2.4745669

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2739046, upper bound: 1.2901331
time: 6.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2808970, upper bound: 1.2902558
time: 9.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.6523466, 9.0279541, -2.1123834, 2.0531931
1: -17.5305214, -13.7629414, -17.4762993, -13.7852192, -3.2060947, 3.0536089
2: -3.2812915, -0.4936304, -3.2660034, -0.5133837, -2.6507850, 2.6621237
3: -10.9046412, -7.9296236, -10.8567801, -7.9611416, -2.9218359, 2.7650270
4: -12.5802679, -9.0147047, -12.6000013, -9.0257072, -3.1108589, 3.1387019
5: -4.9658098, -2.5817404, -4.9100409, -2.6617544, -2.1292305, 2.2218971
6: -3.0887170, -0.4356251, -3.0051947, -0.5469360, -2.3630300, 2.4226718
7: -9.4276257, -5.4237018, -9.4240265, -5.4593987, -3.5040259, 3.3705106
8: -2.6234269, -0.3358107, -2.5963993, -0.3227391, -2.2502403, 2.2302747
9: -4.5114365, -1.7562879, -4.5145378, -1.7673830, -2.5068521, 2.4991338

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2739047, upper bound: 1.2944332
time: 6.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2808970, upper bound: 1.2945562
time: 8.13 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 6.6190963, 9.0374622, 6.6599426, 9.0120716, -2.0835152, 2.0430336
1: -17.4985886, -13.7623196, -17.4670715, -13.8027258, -3.1571932, 3.0344467
2: -3.2717190, -0.5150280, -3.2601101, -0.5296283, -2.6218066, 2.6285410
3: -10.8885679, -7.9413786, -10.8510284, -7.9778361, -2.8776736, 2.7420788
4: -12.5936260, -9.0024395, -12.5209999, -9.0285950, -3.0929222, 3.0733833
5: -4.9584179, -2.5823247, -4.9052739, -2.6745539, -2.1127305, 2.2005630
6: -3.0514343, -0.4269853, -2.9961185, -0.5684438, -2.3095269, 2.3855937
7: -9.5002108, -5.4257102, -9.3222227, -5.4665804, -3.4663911, 3.3214626
8: -2.5898132, -0.3175340, -2.5869083, -0.3497086, -2.2185102, 2.2125075
9: -4.5108137, -1.7477043, -4.4593191, -1.7708025, -2.4591517, 2.4765506

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5773

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2740525, upper bound: 1.2929220
time: 10.67 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2782377, upper bound: 1.2901277
time: 7.49 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2782379, upper bound: 1.2901389
time: 8.32 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 6.5878363, 9.0499210, 6.6551037, 9.0135641, -2.1136343, 2.0660717
1: -17.5419102, -13.7459946, -17.4738770, -13.8005142, -3.2067351, 3.0657859
2: -3.2899222, -0.4797257, -3.2627921, -0.5239124, -2.6477818, 2.6596484
3: -10.9102306, -7.9146261, -10.8545065, -7.9736710, -2.9253969, 2.7807398
4: -12.6525583, -8.9876614, -12.5303135, -9.0268450, -3.1529245, 3.0955796
5: -4.9745345, -2.5696211, -4.9079633, -2.6729436, -2.1351514, 2.2219329
6: -3.0996401, -0.4142132, -3.0038781, -0.5669279, -2.3534222, 2.4253967
7: -9.5286932, -5.3869123, -9.3258781, -5.4603081, -3.5155554, 3.3645425
8: -2.6356559, -0.3081908, -2.5947604, -0.3486171, -2.2462263, 2.2382193
9: -4.5606894, -1.7352264, -4.4674039, -1.7695656, -2.5015621, 2.4980946

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5773

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2847088, upper bound: 1.2945556
time: 13.83 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852298, upper bound: 1.2945551
time: 6.88 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.5967369, 9.0346031, -2.1282046, 2.1282043
1: -17.5305214, -13.7629414, -17.5305214, -13.7629414, -3.2263441, 3.2263446
2: -3.2812915, -0.4936304, -3.2812915, -0.4936304, -2.6645589, 2.6645589
3: -10.9046412, -7.9296236, -10.9046412, -7.9296236, -2.9542904, 2.9542904
4: -12.5802679, -9.0147047, -12.5802679, -9.0147047, -3.1269884, 3.1269870
5: -4.9658098, -2.5817404, -4.9658098, -2.5817404, -2.2786293, 2.2786295
6: -3.0887170, -0.4356251, -3.0887170, -0.4356251, -2.5011144, 2.5011144
7: -9.4276257, -5.4237018, -9.4276257, -5.4237018, -3.5398660, 3.5398660
8: -2.6234269, -0.3358107, -2.6234269, -0.3358107, -2.2607427, 2.2607424
9: -4.5114365, -1.7562879, -4.5114365, -1.7562879, -2.5165257, 2.5165257

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2739103, upper bound: 1.2901278
time: 5.63 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809027, upper bound: 1.2902501
time: 11.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 30.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2739079, upper bound: 1.2807807
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2809002, upper bound: 1.2809055
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2739079, upper bound: 1.2851094
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2809002, upper bound: 1.2852340
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2851091, upper bound: 1.2739100
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2852342, upper bound: 1.2808998
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2782411, upper bound: 1.2808100
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2852344, upper bound: 1.2809332
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2901329, upper bound: 1.2739043
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2902561, upper bound: 1.2808966
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2901329, upper bound: 1.2739052
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2902561, upper bound: 1.2808973
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2901273, upper bound: 1.2782373
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2901292, upper bound: 1.2739378
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2945550, upper bound: 1.2847087
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2945550, upper bound: 1.2852301
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2739046, upper bound: 1.2901331
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2808970, upper bound: 1.2902558
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2739047, upper bound: 1.2944332
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2808970, upper bound: 1.2945562
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2782377, upper bound: 1.2901277
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2782379, upper bound: 1.2901389
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2847088, upper bound: 1.2945556
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2852298, upper bound: 1.2945551
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2739103, upper bound: 1.2901278
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 30.41
Output dim: 0, lower bound: -1.2809027, upper bound: 1.2902501
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.41
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902585
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.41
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945640
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.41
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.088984727859497
rel_dist={0: [-1.2945795095372876, 1.2945797204952658]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9778986
time: 20.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9840132
time: 8.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 29.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 29.08
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9778986
IS_A2, status: Status.UNKNOWN, split count: 1, time: 29.08
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9840132

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.6465387, 9.0227003, -1.8545136, 1.8792005
1: -17.4738808, -13.8005133, -17.4794521, -13.7818222, -2.7337589, 2.8468184
2: -3.2627931, -0.5239117, -3.2686329, -0.5189644, -2.4166198, 2.4108737
3: -10.8545074, -7.9736691, -10.8578558, -7.9554567, -2.5189557, 2.6346593
4: -12.5303125, -9.0268316, -12.5345316, -9.0214090, -2.8220973, 2.8150787
5: -4.9079676, -2.6729443, -4.9376388, -2.6694570, -2.0326710, 1.9228761
6: -3.0038805, -0.5669265, -3.0445609, -0.5626016, -2.2408123, 2.1598985
7: -9.3258791, -5.4602871, -9.3322401, -5.4269423, -3.0712633, 3.1898017
8: -2.5947652, -0.3486171, -2.5980988, -0.3453007, -2.0771899, 2.0838580
9: -4.4674053, -1.7695539, -4.4737139, -1.7593362, -2.3266702, 2.3193996

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840101, upper bound: 0.9752277
time: 7.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840101, upper bound: 0.9778979
time: 9.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.6375170, 9.0313349, -1.9474044, 1.9070358
1: -17.5395622, -13.7612896, -17.4851456, -13.7643452, -2.9453335, 2.8924961
2: -3.2866726, -0.4902350, -3.2758989, -0.5142199, -2.4386067, 2.4539669
3: -10.9079905, -7.9272232, -10.8677921, -7.9381170, -2.7237797, 2.6862383
4: -12.5829248, -8.9888010, -12.5387926, -9.0154057, -2.8883333, 2.8820114
5: -4.9724512, -2.5807762, -4.9653726, -2.6635828, -2.1021028, 2.1373887
6: -3.0983086, -0.4341698, -3.0826273, -0.5545926, -2.3319821, 2.3676996
7: -9.4304581, -5.3878284, -9.3434505, -5.3957191, -3.3186440, 3.2692208
8: -2.6340675, -0.3340945, -2.6018815, -0.3418708, -2.1228929, 2.1026695
9: -4.5135813, -1.7374017, -4.4801064, -1.7481223, -2.3821430, 2.3716447

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9813404, upper bound: 0.9840105
time: 9.63 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840100, upper bound: 0.9840101
time: 10.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 33.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.22
Output dim: 0, lower bound: -0.9840101, upper bound: 0.9752277
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.22
Output dim: 0, lower bound: -0.9840101, upper bound: 0.9778979
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 33.22
Output dim: 0, lower bound: -0.9813404, upper bound: 0.9840105
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 33.22
Output dim: 0, lower bound: -0.9840100, upper bound: 0.9840101

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 6.6572323, 9.0132694, 6.6527719, 9.0218220, -1.8513308, 1.8723581
1: -17.4707909, -13.8010731, -17.4704018, -13.7834816, -2.7290111, 2.8370891
2: -3.2609453, -0.5250452, -3.2632380, -0.5223503, -2.4116359, 2.4044178
3: -10.8533611, -7.9744825, -10.8545094, -7.9578543, -2.5156145, 2.6302080
4: -12.5294170, -9.0356874, -12.5318766, -9.0473156, -2.7951779, 2.8035183
5: -4.9056940, -2.6732683, -4.9309959, -2.6704173, -2.0291085, 1.9149802
6: -3.0005960, -0.5674138, -3.0349584, -0.5640531, -2.2361584, 2.1496775
7: -9.3249321, -5.4725504, -9.3294220, -5.4628162, -3.0343223, 3.1746721
8: -2.5911255, -0.3492079, -2.5874548, -0.3470392, -2.0715060, 2.0724244
9: -4.4666796, -1.7760175, -4.4715724, -1.7782282, -2.3070946, 2.3108513

Time for backsubstitution: 12.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9752279
time: 10.18 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9752277
time: 6.12 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 6.6551056, 9.0135641, 6.6438208, 9.0371237, -1.8686087, 1.8799911
1: -17.4738712, -13.8005123, -17.4818439, -13.7665300, -2.7492766, 2.8472223
2: -3.2627916, -0.5239134, -3.2718420, -0.5084201, -2.4280071, 2.4140360
3: -10.8545055, -7.9736719, -10.8601170, -7.9428644, -2.5343704, 2.6359625
4: -12.5303106, -9.0268536, -12.6042242, -9.0202818, -2.8136411, 2.8709667
5: -4.9079614, -2.6729436, -4.9397116, -2.6582661, -2.0439696, 1.9236422
6: -3.0038753, -0.5669270, -3.0458760, -0.5426083, -2.2613134, 2.1594396
7: -9.3258781, -5.4603238, -9.4304094, -5.4260478, -3.0611997, 3.2423129
8: -2.5947590, -0.3486185, -2.5997195, -0.3194070, -2.0866251, 2.0815496
9: -4.4674058, -1.7695721, -4.5208988, -1.7571706, -2.3230262, 2.3462360

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9778950
time: 9.62 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9778952
time: 6.57 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.6396494, 9.0310383, -1.9405618, 1.9038534
1: -17.5305214, -13.7629414, -17.4820576, -13.7649050, -2.9356046, 2.8877487
2: -3.2812915, -0.4936304, -3.2740564, -0.5153663, -2.4320908, 2.4486928
3: -10.9046412, -7.9296236, -10.8666449, -7.9389424, -2.7193255, 2.6829586
4: -12.5802679, -9.0147047, -12.5378895, -9.0242577, -2.8767185, 2.8550825
5: -4.9658098, -2.5817404, -4.9631014, -2.6639128, -2.0943432, 2.1333897
6: -3.0887170, -0.4356251, -3.0793464, -0.5550866, -2.3219452, 2.3624575
7: -9.4276257, -5.4237018, -9.3424902, -5.4079800, -3.3008084, 3.2322688
8: -2.6234269, -0.3358107, -2.5982437, -0.3424606, -2.1114259, 2.0969930
9: -4.5114365, -1.7562879, -4.4793782, -1.7545953, -2.3722165, 2.3520706

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804933, upper bound: 0.9805037
time: 10.77 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804933, upper bound: 0.9831557
time: 6.62 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 6.5878325, 9.0499220, 6.6375227, 9.0313358, -1.9482224, 1.9211535
1: -17.5419140, -13.7459955, -17.4851360, -13.7643452, -2.9456911, 2.9080067
2: -3.2899241, -0.4797205, -3.2758975, -0.5142226, -2.4417272, 2.4589453
3: -10.9102325, -7.9146242, -10.8677902, -7.9381199, -2.7250671, 2.7020082
4: -12.6525669, -8.9876604, -12.5387907, -9.0154257, -2.9177206, 2.8735509
5: -4.9745350, -2.5696204, -4.9653664, -2.6635833, -2.1027741, 2.1394920
6: -3.0996435, -0.4142132, -3.0826206, -0.5545950, -2.3316240, 2.3722961
7: -9.5286961, -5.3869095, -9.3434505, -5.3957534, -3.3320084, 3.2591987
8: -2.6356602, -0.3081894, -2.6018763, -0.3418703, -2.1205983, 2.1134288
9: -4.5606918, -1.7352251, -4.4801068, -1.7481396, -2.3890216, 2.3679714

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9835848, upper bound: 0.9835857
time: 10.86 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840089, upper bound: 0.9840114
time: 11.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 35.32 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9752279
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9752277
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9778950
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9779490, upper bound: 0.9778952
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9804933, upper bound: 0.9805037
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9804933, upper bound: 0.9831557
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9835848, upper bound: 0.9835857
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 35.32
Output dim: 0, lower bound: -0.9840089, upper bound: 0.9840114

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 6.6572323, 9.0132694, 6.6613379, 9.0126896, -1.8419442, 1.8382807
1: -17.4707909, -13.8010731, -17.4648304, -13.8021736, -2.7095370, 2.7045574
2: -3.2609453, -0.5250452, -3.2573876, -0.5272779, -2.4059339, 2.4044504
3: -10.8533611, -7.9744825, -10.8511620, -7.9760609, -2.4966736, 2.4955604
4: -12.5294170, -9.0356874, -12.5276699, -9.0527420, -2.7827086, 2.7980704
5: -4.9056940, -2.6732683, -4.9013195, -2.6738970, -1.8893871, 1.8850737
6: -3.0005960, -0.5674138, -2.9942713, -0.5683670, -2.1109505, 2.1054246
7: -9.3249321, -5.4725504, -9.3230848, -5.4961648, -3.0023346, 3.0242023
8: -2.5911255, -0.3492079, -2.5841184, -0.3503575, -2.0682049, 2.0624461
9: -4.4666796, -1.7760175, -4.4652653, -1.7884412, -2.2925100, 2.3035498

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9775293, upper bound: 0.9748058
time: 11.68 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779480, upper bound: 0.9752268
time: 8.86 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 6.6572323, 9.0132694, 6.5989819, 9.0345812, -1.8643656, 1.9203981
1: -17.4707909, -13.8010731, -17.5300007, -13.7630663, -2.7507114, 2.8989797
2: -3.2609453, -0.5250452, -3.2811546, -0.4952123, -2.4392319, 2.4217858
3: -10.8533611, -7.9744825, -10.9042959, -7.9304504, -2.5443482, 2.6790853
4: -12.5294170, -9.0356874, -12.5800629, -9.0186205, -2.8157468, 2.8530231
5: -4.9056940, -2.6732683, -4.9656525, -2.5824852, -2.0765204, 1.9440265
6: -3.0005960, -0.5674138, -3.0881839, -0.4364614, -2.2820940, 2.1812701
7: -9.3249321, -5.4725504, -9.4270048, -5.4251251, -3.0773239, 3.2370324
8: -2.5911255, -0.3492079, -2.6205978, -0.3358879, -2.0842133, 2.1010575
9: -4.4666796, -1.7760175, -4.5113444, -1.7575762, -2.3256106, 2.3470025

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9775270, upper bound: 0.9748060
time: 6.10 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779480, upper bound: 0.9752267
time: 7.29 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 6.6551056, 9.0135641, 6.6523466, 9.0279541, -1.8592031, 1.8459451
1: -17.4738712, -13.8005123, -17.4762993, -13.7852192, -2.7298021, 2.7147131
2: -3.2627916, -0.5239134, -3.2660034, -0.5133837, -2.4222851, 2.4140913
3: -10.8545055, -7.9736719, -10.8567801, -7.9611416, -2.5153913, 2.5013893
4: -12.5303106, -9.0268536, -12.6000013, -9.0257072, -2.8011775, 2.8654819
5: -4.9079614, -2.6729436, -4.9100409, -2.6617544, -1.9044528, 1.8937409
6: -3.0038753, -0.5669270, -3.0051947, -0.5469360, -2.1363668, 2.1151876
7: -9.3258781, -5.4603238, -9.4240265, -5.4593987, -3.0292130, 3.0922189
8: -2.5947590, -0.3486185, -2.5963993, -0.3227391, -2.0832930, 2.0716000
9: -4.4674058, -1.7695721, -4.5145378, -1.7673830, -2.3084636, 2.3395717

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9775293, upper bound: 0.9774735
time: 8.83 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779507, upper bound: 0.9778951
time: 7.89 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 6.6551056, 9.0135641, 6.5900803, 9.0499001, -1.8802047, 1.9280568
1: -17.4738712, -13.8005123, -17.5413933, -13.7461166, -2.7709732, 2.9090681
2: -3.2627916, -0.5239134, -3.2897859, -0.4813030, -2.4501805, 2.4314647
3: -10.8545055, -7.9736719, -10.9098873, -7.9154501, -2.5631351, 2.6848249
4: -12.5303106, -9.0268536, -12.6523581, -8.9915752, -2.8342257, 2.8938458
5: -4.9079614, -2.6729436, -4.9743786, -2.5703549, -2.0826287, 1.9527347
6: -3.0038753, -0.5669270, -3.0991123, -0.4150491, -2.2919335, 2.1910477
7: -9.3258781, -5.4603238, -9.5280743, -5.3883324, -3.1042132, 3.2682359
8: -2.5947590, -0.3486185, -2.6328292, -0.3082666, -2.0969219, 2.1102357
9: -4.4674058, -1.7695721, -4.5606008, -1.7365143, -2.3415575, 2.3638630

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9775270, upper bound: 0.9774756
time: 6.11 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779480, upper bound: 0.9778941
time: 6.59 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 6.6010947, 9.0336723, 6.6503663, 9.0252275, -1.9282236, 1.8920026
1: -17.5277023, -13.7647800, -17.4743042, -13.7701406, -2.9205132, 2.8774176
2: -3.2797866, -0.4981469, -3.2682927, -0.5262885, -2.4191456, 2.4355607
3: -10.9029045, -7.9330149, -10.8622589, -7.9475403, -2.7056441, 2.6722751
4: -12.5746183, -9.0157642, -12.5239019, -9.0294514, -2.8658805, 2.8398132
5: -4.9643173, -2.5825369, -4.9590850, -2.6674891, -2.0869422, 2.1264961
6: -3.0833869, -0.4366140, -3.0672979, -0.5611339, -2.3096380, 2.3486960
7: -9.4253578, -5.4281111, -9.3317051, -5.4177008, -3.2880821, 3.2166634
8: -2.6171088, -0.3363714, -2.5843072, -0.3469586, -2.0992899, 2.0834017
9: -4.5067630, -1.7568541, -4.4686499, -1.7593555, -2.3608615, 2.3396049

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9800705, upper bound: 0.9800880
time: 6.80 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804922, upper bound: 0.9805030
time: 10.09 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346041, 6.6396532, 9.0310364, -1.9381356, 1.9005623
1: -17.5305214, -13.7630138, -17.4820499, -13.7650909, -2.9321876, 2.8775244
2: -3.2812905, -0.4936340, -3.2740536, -0.5153776, -2.4272299, 2.4465830
3: -10.9046402, -7.9296255, -10.8666449, -7.9389477, -2.7193136, 2.6837225
4: -12.5802650, -9.0147066, -12.5378799, -9.0242615, -2.8767142, 2.8548460
5: -4.9658098, -2.5818424, -4.9631009, -2.6641760, -2.0931902, 2.1344614
6: -3.0887139, -0.4356260, -3.0793390, -0.5550880, -2.3219399, 2.3575428
7: -9.4276228, -5.4237056, -9.3424854, -5.4079847, -3.2985659, 3.2322631
8: -2.6234226, -0.3358102, -2.5982361, -0.3424616, -2.1088152, 2.0907331
9: -4.5114346, -1.7562867, -4.4793701, -1.7545969, -2.3707547, 2.3482988

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9800732, upper bound: 0.9827437
time: 7.08 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804951, upper bound: 0.9831548
time: 6.46 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: 6.5929017, 9.0466766, 6.6393762, 9.0302391, -1.9425476, 1.9165285
1: -17.5381908, -13.7622681, -17.4838963, -13.7702875, -2.9359493, 2.8903761
2: -3.2840815, -0.4872599, -3.2738066, -0.5169828, -2.4323506, 2.4498465
3: -10.9068356, -7.9329386, -10.8666601, -7.9445057, -2.7143497, 2.6824870
4: -12.6462040, -8.9930458, -12.5366211, -9.0178928, -2.9064415, 2.8661580
5: -4.9709973, -2.5783668, -4.9641590, -2.6665239, -2.0947461, 2.1282735
6: -3.0912049, -0.4166856, -3.0797377, -0.5554152, -2.3229823, 2.3668187
7: -9.5139313, -5.3940210, -9.3385191, -5.3986211, -3.3149309, 3.2480392
8: -2.6305223, -0.3136716, -2.6001630, -0.3437099, -2.1136646, 2.1057169
9: -4.5447464, -1.7383757, -4.4747753, -1.7493062, -2.3714707, 2.3599370

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827352, upper bound: 0.9800880
time: 5.84 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827380, upper bound: 0.9827409
time: 6.16 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: 6.5718422, 9.0536928, 6.6375265, 9.0313339, -1.9606953, 1.9245310
1: -17.5871506, -13.7353020, -17.4851322, -13.7643681, -2.9683352, 2.9149737
2: -3.3006895, -0.4715343, -3.2758896, -0.5142300, -2.4488077, 2.4694452
3: -10.9477825, -7.9032550, -10.8677855, -7.9381404, -2.7521114, 2.7115679
4: -12.6625805, -8.9738531, -12.5387850, -9.0154333, -2.9283893, 2.8902717
5: -4.9959354, -2.5638576, -4.9653640, -2.6635942, -2.1250148, 2.1441202
6: -3.1046238, -0.4009137, -3.0826144, -0.5545969, -2.3387928, 2.3801792
7: -9.5372829, -5.3442678, -9.3434420, -5.3957605, -3.3391452, 3.3026004
8: -2.6459656, -0.3026261, -2.6018710, -0.3418784, -2.1300559, 2.1186860
9: -4.5755863, -1.7096874, -4.4800882, -1.7481428, -2.4003766, 2.3767352

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831555, upper bound: 0.9805057
time: 7.94 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831555, upper bound: 0.9831559
time: 6.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.02 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9775293, upper bound: 0.9748058
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9779480, upper bound: 0.9752268
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9775270, upper bound: 0.9748060
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9779480, upper bound: 0.9752267
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9775293, upper bound: 0.9774735
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9779507, upper bound: 0.9778951
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9775270, upper bound: 0.9774756
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9779480, upper bound: 0.9778941
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9800705, upper bound: 0.9800880
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9804922, upper bound: 0.9805030
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9800732, upper bound: 0.9827437
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9804951, upper bound: 0.9831548
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9827352, upper bound: 0.9800880
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9827380, upper bound: 0.9827409
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9831555, upper bound: 0.9805057
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 0, lower bound: -0.9831555, upper bound: 0.9831559

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 6.6590576, 9.0121937, 6.6663141, 9.0095310, -1.8374114, 1.8327014
1: -17.4695625, -13.8070183, -17.4611282, -13.8184547, -2.6919150, 2.6948214
2: -3.2588429, -0.5278479, -3.2513504, -0.5348618, -2.3969226, 2.3950999
3: -10.8522367, -7.9808702, -10.8477745, -7.9943905, -2.4771194, 2.4849074
4: -12.5272560, -9.0381536, -12.5213718, -9.0581350, -2.7753329, 2.7867184
5: -4.9044743, -2.6762056, -4.8976941, -2.6826291, -1.8779860, 1.8758583
6: -2.9977028, -0.5682268, -2.9858088, -0.5708137, -2.1054878, 2.0963709
7: -9.3199997, -5.4754057, -9.3083344, -5.5032473, -2.9911790, 3.0075955
8: -2.5893803, -0.3510499, -2.5789528, -0.3558526, -2.0604925, 2.0554807
9: -4.4613605, -1.7771842, -4.4493766, -1.7915896, -2.2844048, 2.2858365

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9774046, upper bound: 0.9747311
time: 5.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9774017, upper bound: 0.9748587
time: 5.61 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 6.6572371, 9.0132656, 6.6450500, 9.0163832, -1.8452682, 1.8551905
1: -17.4707832, -13.8010941, -17.5101814, -13.7914963, -2.7164865, 2.7508960
2: -3.2609391, -0.5250540, -3.2677677, -0.5189903, -2.4162641, 2.4117107
3: -10.8533592, -7.9745045, -10.8887739, -7.9648271, -2.5059843, 2.5334473
4: -12.5294094, -9.0356932, -12.5375853, -9.0389290, -2.7994394, 2.8159175
5: -4.9056926, -2.6732788, -4.9228067, -2.6681275, -1.8940215, 1.9073207
6: -3.0005891, -0.5674162, -2.9992304, -0.5550051, -2.1256936, 2.1129260
7: -9.3249245, -5.4725571, -9.3315830, -5.4539266, -3.0451517, 3.0318899
8: -2.5911202, -0.3492146, -2.5945492, -0.3448281, -2.0731778, 2.0728254
9: -4.4666600, -1.7760212, -4.4803209, -1.7628860, -2.3180287, 2.3150973

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9771109, upper bound: 0.9717786
time: 6.13 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9771082, upper bound: 0.9744394
time: 5.51 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 6.6622081, 9.0101070, 6.6008368, 9.0334797, -1.8587737, 1.9158629
1: -17.4670906, -13.8173523, -17.5287647, -13.7690086, -2.7409725, 2.8813639
2: -3.2549171, -0.5326327, -3.2790871, -0.4979903, -2.4298787, 2.4127533
3: -10.8499746, -7.9928174, -10.9031677, -7.9368868, -2.5336943, 2.6595693
4: -12.5231190, -9.0410795, -12.5778761, -9.0210857, -2.8044119, 2.8456483
5: -4.9020700, -2.6820009, -4.9644594, -2.5854251, -2.0679007, 1.9324927
6: -2.9921308, -0.5698595, -3.0853021, -0.4372835, -2.2734709, 2.1757624
7: -9.3101807, -5.4796314, -9.4220724, -5.4279885, -3.0607109, 3.2251253
8: -2.5859609, -0.3547034, -2.6188703, -0.3377247, -2.0772481, 2.0933611
9: -4.4507899, -1.7791653, -4.5060072, -1.7587427, -2.3080802, 2.3382232

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827347, upper bound: 0.9713088
time: 5.82 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827347, upper bound: 0.9739659
time: 5.66 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 6.6409516, 9.0169640, 6.5989866, 9.0345783, -1.8812702, 1.9237173
1: -17.5161362, -13.7903938, -17.5299988, -13.7630863, -2.7970457, 2.9059300
2: -3.2713428, -0.5167483, -3.2811475, -0.4952213, -2.4464893, 2.4322064
3: -10.8909702, -7.9632454, -10.9042921, -7.9304676, -2.5822334, 2.6884806
4: -12.5393343, -9.0218754, -12.5800562, -9.0186281, -2.8336048, 2.8697548
5: -4.9271793, -2.6674976, -4.9656496, -2.5824966, -2.0834026, 1.9486084
6: -3.0055587, -0.5540533, -3.0881770, -0.4364638, -2.2893465, 2.1892526
7: -9.3334312, -5.4303126, -9.4269962, -5.4251328, -3.0850143, 3.2504892
8: -2.6015491, -0.3436775, -2.6205921, -0.3358936, -2.0945888, 2.1060863
9: -4.4817305, -1.7504667, -4.5113258, -1.7575800, -2.3371019, 2.3519757

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831550, upper bound: 0.9717260
time: 10.36 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831550, upper bound: 0.9743862
time: 5.71 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 6.6569324, 9.0124893, 6.6573234, 9.0247841, -1.8546550, 1.8403628
1: -17.4726467, -13.8064594, -17.4725952, -13.8015013, -2.7121792, 2.7049751
2: -3.2606919, -0.5267168, -3.2599447, -0.5209533, -2.4132700, 2.4047368
3: -10.8533802, -7.9800577, -10.8533945, -7.9794688, -2.4958286, 2.4907274
4: -12.5281487, -9.0293207, -12.5936823, -9.0311022, -2.7938027, 2.8541434
5: -4.9067421, -2.6758814, -4.9064183, -2.6704903, -1.8930411, 1.8845270
6: -3.0009818, -0.5677404, -2.9967294, -0.5493813, -2.1308570, 2.1061292
7: -9.3209448, -5.4631777, -9.4092741, -5.4664798, -3.0180559, 3.0752530
8: -2.5930142, -0.3504605, -2.5912337, -0.3282342, -2.0756001, 2.0646348
9: -4.4620843, -1.7707387, -4.4986444, -1.7705318, -2.3003559, 2.3218286

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9774017, upper bound: 0.9774016
time: 10.99 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9774046, upper bound: 0.9775268
time: 10.08 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 6.6551113, 9.0135622, 6.6360765, 9.0316820, -1.8625393, 1.8628421
1: -17.4738712, -13.8005342, -17.5216370, -13.7745428, -2.7367516, 2.7610407
2: -3.2627859, -0.5239222, -3.2763991, -0.5050498, -2.4326668, 2.4213185
3: -10.8545027, -7.9736915, -10.8943834, -7.9499111, -2.5247183, 2.5392611
4: -12.5303049, -9.0268593, -12.6099205, -9.0118961, -2.8179102, 2.8761170
5: -4.9079599, -2.6729546, -4.9315271, -2.6559753, -1.9090958, 1.9159770
6: -3.0038688, -0.5669289, -3.0101609, -0.5335774, -2.1510553, 2.1226974
7: -9.3258705, -5.4603305, -9.4325361, -5.4171572, -3.0720272, 3.0992899
8: -2.5947547, -0.3486257, -2.6068192, -0.3172121, -2.0885253, 2.0819807
9: -4.4673862, -1.7695755, -4.5296307, -1.7418368, -2.3340149, 2.3509743

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9771082, upper bound: 0.9744462
time: 5.72 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9771082, upper bound: 0.9771073
time: 6.03 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 6.6600800, 9.0104027, 6.5919361, 9.0487947, -1.8745570, 1.9235185
1: -17.4701767, -13.8167953, -17.5401611, -13.7520599, -2.7612357, 2.8914537
2: -3.2567692, -0.5315011, -3.2876980, -0.4840655, -2.4407363, 2.4224248
3: -10.8511209, -7.9920063, -10.9087591, -7.9218249, -2.5524807, 2.6653070
4: -12.5240145, -9.0322456, -12.6501722, -8.9940405, -2.8228931, 2.8862772
5: -4.9043379, -2.6816781, -4.9731855, -2.5732961, -2.0740023, 1.9412000
6: -2.9954093, -0.5693731, -3.0962296, -0.4158731, -2.2832942, 2.1855414
7: -9.3111248, -5.4674034, -9.5231400, -5.3911982, -3.0875969, 3.2563350
8: -2.5895948, -0.3541131, -2.6311030, -0.3101044, -2.0900021, 2.1025379
9: -4.4515152, -1.7727196, -4.5552626, -1.7376812, -2.3240256, 2.3550823

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827347, upper bound: 0.9739788
time: 5.94 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827347, upper bound: 0.9766358
time: 5.57 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 6.6388292, 9.0172596, 6.5900860, 9.0498972, -1.8927693, 1.9313761
1: -17.5192146, -13.7898369, -17.5413895, -13.7461376, -2.8064947, 2.9160194
2: -3.2731984, -0.5156103, -3.2897811, -0.4813123, -2.4571440, 2.4418905
3: -10.8921146, -7.9624338, -10.9098835, -7.9154696, -2.6010180, 2.6942213
4: -12.5402288, -9.0130405, -12.6523523, -8.9915810, -2.8520823, 2.9027681
5: -4.9294443, -2.6671729, -4.9743743, -2.5703654, -2.0895123, 1.9573171
6: -3.0088406, -0.5535665, -3.0991049, -0.4150519, -2.2991891, 2.1990337
7: -9.3343782, -5.4180875, -9.5280666, -5.3883410, -3.1119051, 3.2816937
8: -2.6051807, -0.3430891, -2.6328244, -0.3082728, -2.1063542, 2.1152656
9: -4.4824524, -1.7440259, -4.5605826, -1.7365184, -2.3530488, 2.3688362

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831550, upper bound: 0.9743933
time: 9.82 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831550, upper bound: 0.9770548
time: 5.34 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 6.6061592, 9.0304403, 6.6522169, 9.0241318, -1.9225559, 1.8873963
1: -17.5239792, -13.7810535, -17.4730606, -13.7760859, -2.9107695, 2.8597884
2: -3.2738914, -0.5057054, -3.2661943, -0.5290517, -2.4097881, 2.4264810
3: -10.8995066, -7.9514008, -10.8611298, -7.9539137, -2.6949286, 2.6527658
4: -12.5682611, -9.0211496, -12.5217371, -9.0319195, -2.8545351, 2.8324127
5: -4.9607735, -2.5912714, -4.9578733, -2.6704290, -2.0789385, 2.1152842
6: -3.0749488, -0.4390903, -3.0644138, -0.5619569, -2.3010049, 2.3432813
7: -9.4106054, -5.4352198, -9.3267727, -5.4205651, -3.2710028, 3.2055035
8: -2.6119776, -0.3418541, -2.5825925, -0.3487959, -2.0923634, 2.0756764
9: -4.4908333, -1.7600034, -4.4633288, -1.7605218, -2.3433247, 2.3315790

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9799433, upper bound: 0.9799628
time: 12.85 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9799433, upper bound: 0.9800877
time: 5.56 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 6.5851107, 9.0374241, 6.6503725, 9.0252237, -1.9407172, 1.8953624
1: -17.5729465, -13.7540922, -17.4743004, -13.7701616, -2.9468198, 2.8843818
2: -3.2904801, -0.4900091, -3.2682867, -0.5262975, -2.4262738, 2.4461563
3: -10.9404640, -7.9217157, -10.8622532, -7.9475603, -2.7342691, 2.6818118
4: -12.5846243, -9.0019569, -12.5238972, -9.0294571, -2.8837175, 2.8565340
5: -4.9857178, -2.5767808, -4.9590831, -2.6675005, -2.1091909, 2.1311095
6: -3.0883589, -0.4233088, -3.0672903, -0.5611367, -2.3167944, 2.3566473
7: -9.4338684, -5.3854651, -9.3316975, -5.4177084, -3.2951870, 3.2600708
8: -2.6274486, -0.3308063, -2.5843024, -0.3469653, -2.1087484, 2.0884180
9: -4.5216393, -1.7313044, -4.4686298, -1.7593576, -2.3722036, 2.3482785

Time for backsubstitution: 12.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804922, upper bound: 0.9778430
time: 10.00 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804922, upper bound: 0.9805030
time: 9.20 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 6.6018033, 9.0313683, 6.6415067, 9.0299406, -1.9324620, 1.8959513
1: -17.5267982, -13.7792883, -17.4808102, -13.7710342, -2.9224405, 2.8598909
2: -3.2754135, -0.5011891, -3.2719631, -0.5181377, -2.4178696, 2.4374838
3: -10.9012432, -7.9480000, -10.8655128, -7.9453273, -2.7085972, 2.6642070
4: -12.5739059, -9.0200901, -12.5357122, -9.0267277, -2.8653755, 2.8474541
5: -4.9622679, -2.5905757, -4.9618912, -2.6671162, -2.0851812, 2.1232464
6: -3.0802770, -0.4380999, -3.0764554, -0.5559096, -2.3133073, 2.3521075
7: -9.4128742, -5.4308128, -9.3375540, -5.4108515, -3.2814770, 3.2211046
8: -2.6182885, -0.3412938, -2.5965242, -0.3443003, -2.1018729, 2.0830045
9: -4.4954915, -1.7594361, -4.4740400, -1.7557635, -2.3532031, 2.3402665

Time for backsubstitution: 12.69 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.9054350852966309
rel_dist={0: [-0.9840186515659699, 0.9840187223222951]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742867
time: 6.76 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742859, upper bound: 0.8742891
time: 10.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.71 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 17.71
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742867
IS_B2, status: Status.UNKNOWN, split count: 1, time: 17.71
Output dim: 0, lower bound: -0.8742859, upper bound: 0.8742891

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: 6.6483841, 9.0208206, 6.6551013, 9.0135641, -1.7899332, 1.7911229
1: -17.4782696, -13.7856483, -17.4738808, -13.8005133, -2.6196427, 2.6310878
2: -3.2672892, -0.5199802, -3.2627931, -0.5239117, -2.3436890, 2.3435085
3: -10.8565407, -7.9592094, -10.8545074, -7.9736691, -2.4299316, 2.4428222
4: -12.5336456, -9.0226068, -12.5303125, -9.0268316, -2.7304788, 2.7368698
5: -4.9315715, -2.6704021, -4.9079676, -2.6729443, -1.8614388, 1.8402200
6: -3.0362353, -0.5638542, -3.0038805, -0.5669265, -2.1008332, 2.0689964
7: -9.3304529, -5.4337687, -9.3258791, -5.4602871, -2.9613743, 2.9822497
8: -2.5973587, -0.3460083, -2.5947652, -0.3486171, -2.0295668, 2.0295727
9: -4.4723735, -1.7615666, -4.4674053, -1.7695539, -2.2686400, 2.2728472

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698165, upper bound: 0.8722829
time: 7.20 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698165, upper bound: 0.8742843
time: 7.22 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: 6.6375189, 9.0313349, 6.5904884, 9.0354805, -1.8454709, 1.8851538
1: -17.4851456, -13.7643452, -17.5395622, -13.7612896, -2.7954140, 2.8484926
2: -3.2758975, -0.5142211, -3.2866726, -0.4902350, -2.3812413, 2.3659151
3: -10.8677931, -7.9381194, -10.9079905, -7.9272232, -2.6109838, 2.6484599
4: -12.5387926, -9.0154037, -12.5829248, -8.9888010, -2.7976289, 2.8057685
5: -4.9653697, -2.6635838, -4.9724512, -2.5807762, -2.0907946, 2.0558095
6: -3.0826256, -0.5545936, -3.0983086, -0.4341698, -2.3228676, 2.2864563
7: -9.3434505, -5.3957224, -9.4304581, -5.3878284, -3.1883574, 3.2355382
8: -2.6018820, -0.3418722, -2.6340675, -0.3340945, -2.0553894, 2.0750735
9: -4.4801068, -1.7481221, -4.5135813, -1.7374017, -2.3200216, 2.3314006

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742864, upper bound: 0.8722809
time: 6.38 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742836, upper bound: 0.8742837
time: 9.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 30.70 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 30.70
Output dim: 0, lower bound: -0.8698165, upper bound: 0.8722829
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 30.70
Output dim: 0, lower bound: -0.8698165, upper bound: 0.8742843
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 30.70
Output dim: 0, lower bound: -0.8742864, upper bound: 0.8722809
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 30.70
Output dim: 0, lower bound: -0.8742836, upper bound: 0.8742837

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: 6.6483908, 9.0208187, 6.6523466, 9.0279541, -1.8040121, 1.7914782
1: -17.4782562, -13.7856503, -17.4762993, -13.7852192, -2.6351576, 2.6307139
2: -3.2672849, -0.5199823, -3.2660034, -0.5133837, -2.3550706, 2.3465421
3: -10.8565378, -7.9592104, -10.8567801, -7.9611416, -2.4453135, 2.4440637
4: -12.5336437, -9.0226355, -12.6000013, -9.0257072, -2.7187796, 2.7848165
5: -4.9315672, -2.6704030, -4.9100409, -2.6617544, -1.8729239, 1.8405042
6: -3.0362296, -0.5638547, -3.0051947, -0.5469360, -2.1194658, 2.0678942
7: -9.3304491, -5.4338078, -9.4240265, -5.4593987, -2.9472055, 3.0250671
8: -2.5973530, -0.3460097, -2.5963993, -0.3227391, -2.0383563, 2.0259261
9: -4.4723716, -1.7615867, -4.5145378, -1.7673830, -2.2629366, 2.2968647

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698165, upper bound: 0.8699027
time: 7.00 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698134, upper bound: 0.8742841
time: 6.56 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: 6.6405210, 9.0309134, 6.5967369, 9.0346031, -1.8413751, 1.8781749
1: -17.4807911, -13.7651367, -17.5305214, -13.7629414, -2.7893820, 2.8385367
2: -3.2733006, -0.5158434, -3.2812915, -0.4936304, -2.3751948, 2.3589585
3: -10.8661757, -7.9392834, -10.9046412, -7.9296236, -2.6071739, 2.6437249
4: -12.5375166, -9.0278854, -12.5802679, -9.0147047, -2.7703247, 2.7905135
5: -4.9621696, -2.6640487, -4.9658098, -2.5817404, -2.0857315, 2.0479028
6: -3.0780029, -0.5552912, -3.0887170, -0.4356251, -2.3162513, 2.2762201
7: -9.3420906, -5.4130039, -9.4276257, -5.4237018, -3.1510086, 3.2125070
8: -2.5967550, -0.3427033, -2.6234269, -0.3358107, -2.0482073, 2.0633214
9: -4.4790764, -1.7572374, -4.5114365, -1.7562879, -2.3001575, 2.3187506

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8738254, upper bound: 0.8719633
time: 9.69 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742820, upper bound: 0.8722790
time: 6.52 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: 6.6375237, 9.0313320, 6.5878325, 9.0499220, -1.8595877, 1.8855033
1: -17.4851360, -13.7643471, -17.5419140, -13.7459955, -2.8109226, 2.8480496
2: -3.2758942, -0.5142224, -3.2899241, -0.4797205, -2.3862185, 2.3688803
3: -10.8677902, -7.9381227, -10.9102325, -7.9146242, -2.6267524, 2.6496067
4: -12.5387897, -9.0154295, -12.6525669, -8.9876604, -2.7859211, 2.8298123
5: -4.9653625, -2.6635838, -4.9745350, -2.5696204, -2.0928984, 2.0559921
6: -3.0826192, -0.5545945, -3.0996435, -0.4142132, -2.3274641, 2.2854507
7: -9.3434486, -5.3957629, -9.5286961, -5.3869095, -3.1742334, 3.2489023
8: -2.6018744, -0.3418732, -2.6356602, -0.3081894, -2.0655842, 2.0714235
9: -4.4801054, -1.7481431, -4.5606918, -1.7352251, -2.3142757, 2.3382790

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8739661, upper bound: 0.8738265
time: 9.64 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742821, upper bound: 0.8742833
time: 21.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 45.87 seconds
IS_B1_B2_A1, status: Status.VERIFIED, split count: 3, time: 45.87
Output dim: 0, lower bound: -0.8698165, upper bound: 0.8699027
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 45.87
Output dim: 0, lower bound: -0.8698134, upper bound: 0.8742841
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 45.87
Output dim: 0, lower bound: -0.8738254, upper bound: 0.8719633
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 45.87
Output dim: 0, lower bound: -0.8742820, upper bound: 0.8722790
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 45.87
Output dim: 0, lower bound: -0.8739661, upper bound: 0.8738265
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 45.87
Output dim: 0, lower bound: -0.8742821, upper bound: 0.8742833

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: 6.5955954, 9.0353889, 6.6523466, 9.0279541, -1.8387308, 1.8063548
1: -17.5388050, -13.7615910, -17.4762993, -13.7852192, -2.6964788, 2.6556168
2: -3.2843299, -0.4933312, -3.2660034, -0.5133837, -2.3720937, 2.3738606
3: -10.9007874, -7.9289107, -10.8567801, -7.9611416, -2.4935808, 2.4757848
4: -12.5822744, -8.9959431, -12.6000013, -9.0257072, -2.7681694, 2.8042510
5: -4.9719238, -2.5834897, -4.9100409, -2.6617544, -1.8928914, 1.8808472
6: -3.0972273, -0.4384198, -3.0051947, -0.5469360, -2.1368446, 2.1108463
7: -9.4247837, -5.3903642, -9.4240265, -5.4593987, -3.0052109, 3.0418975
8: -2.6283717, -0.3344965, -2.5963993, -0.3227391, -2.0574098, 2.0384769
9: -4.5129652, -1.7415240, -4.5145378, -1.7673830, -2.3003001, 2.3091793

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_B1_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8694556, upper bound: 0.8738248
time: 6.00 seconds

## Relational analysis of IS_B1_B2_A2_B2

### Relational analysis result of IS_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698127, upper bound: 0.8679022
time: 157.23 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: 6.6455812, 9.0276966, 6.5993738, 9.0330324, -1.8352404, 1.8729540
1: -17.4770432, -13.7814140, -17.5287552, -13.7713833, -2.7771692, 2.8203521
2: -3.2672718, -0.5233666, -3.2783594, -0.4975641, -2.3646860, 2.3489635
3: -10.8627739, -7.9575796, -10.9030266, -7.9387670, -2.5937762, 2.6236057
4: -12.5312033, -9.0332756, -12.5771580, -9.0181713, -2.7582574, 2.7817419
5: -4.9585810, -2.6727905, -4.9641061, -2.5859339, -2.0757446, 2.0361345
6: -3.0695655, -0.5577612, -3.0846100, -0.4368029, -2.3072743, 2.2697093
7: -9.3273430, -5.4201202, -9.4205828, -5.4277592, -3.1332998, 3.1983314
8: -2.5916028, -0.3481889, -2.6209679, -0.3384337, -2.0403342, 2.0549276
9: -4.4631572, -1.7603877, -4.5038152, -1.7579501, -2.2823095, 2.3075812

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8707569, upper bound: 0.8708222
time: 9.24 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726866, upper bound: 0.8708217
time: 6.12 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: 6.6243010, 9.0346518, 6.5967426, 9.0346003, -1.8582730, 1.8814491
1: -17.5260220, -13.7544613, -17.5305176, -13.7629700, -2.8355665, 2.8442359
2: -3.2837882, -0.5076280, -3.2812850, -0.4936408, -2.3821230, 2.3692195
3: -10.9037762, -7.9280858, -10.9046383, -7.9296455, -2.6449804, 2.6519051
4: -12.5474606, -9.0141430, -12.5802622, -9.0147133, -2.7877293, 2.8071742
5: -4.9836330, -2.6582932, -4.9658074, -2.5817528, -2.0925980, 2.0521350
6: -3.0829618, -0.5421205, -3.0887079, -0.4356279, -2.3234000, 2.2907255
7: -9.3506403, -5.3705735, -9.4276142, -5.4237103, -3.1574283, 3.2258584
8: -2.6071844, -0.3371854, -2.6234217, -0.3358188, -2.0585675, 2.0683448
9: -4.4939799, -1.7317162, -4.5114150, -1.7562913, -2.3108535, 2.3237033

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8712155, upper bound: 0.8711418
time: 5.58 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8731462, upper bound: 0.8711425
time: 9.00 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: 6.6401596, 9.0297718, 6.5929017, 9.0466766, -1.8543015, 1.8793709
1: -17.4833565, -13.7727890, -17.5381908, -13.7622681, -2.7927122, 2.8358707
2: -3.2729354, -0.5181379, -3.2840815, -0.4872599, -2.3761292, 2.3584647
3: -10.8661728, -7.9472108, -10.9068356, -7.9329386, -2.6065931, 2.6362333
4: -12.5357018, -9.0188999, -12.6462040, -8.9930458, -2.7771316, 2.8178289
5: -4.9636369, -2.6677802, -4.9709973, -2.5783668, -2.0809717, 2.0465922
6: -3.0785108, -0.5557709, -3.0912049, -0.4166856, -2.3208413, 2.2764225
7: -9.3364077, -5.3998232, -9.5139313, -5.3940210, -3.1610379, 3.2307467
8: -2.5994239, -0.3444963, -2.6305223, -0.3136716, -2.0571904, 2.0636091
9: -4.4724989, -1.7498055, -4.5447464, -1.7383757, -2.3038568, 2.3203743

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8708995, upper bound: 0.8726874
time: 8.91 seconds

## Relational analysis of IS_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8728255, upper bound: 0.8726871
time: 8.14 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: 6.6375294, 9.0313320, 6.5718479, 9.0536890, -1.8628874, 1.8979702
1: -17.4851322, -13.7643709, -17.5870018, -13.7353020, -2.8166389, 2.8692749
2: -3.2758884, -0.5142341, -3.3006537, -0.4715408, -2.3966551, 2.3759120
3: -10.8677864, -7.9381442, -10.9477797, -7.9032869, -2.6350822, 2.6757407
4: -12.5387850, -9.0154381, -12.6625748, -8.9739227, -2.8025665, 2.8400381
5: -4.9653616, -2.6635957, -4.9959173, -2.5638599, -2.0971050, 2.0782123
6: -3.0826106, -0.5545974, -3.1046128, -0.4011006, -2.3352027, 2.2925231
7: -9.3434391, -5.3957729, -9.5372581, -5.3442779, -3.2176218, 3.2547998
8: -2.6018696, -0.3418818, -2.6459389, -0.3026342, -2.0708332, 2.0808518
9: -4.4800839, -1.7481465, -4.5755558, -1.7097167, -2.3218393, 2.3488328

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8712182, upper bound: 0.8731469
time: 9.27 seconds

## Relational analysis of IS_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8731462, upper bound: 0.8731481
time: 7.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.60 seconds
IS_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8694556, upper bound: 0.8738248
IS_B1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8698127, upper bound: 0.8679022
IS_B2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8707569, upper bound: 0.8708222
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8726866, upper bound: 0.8708217
IS_B2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8712155, upper bound: 0.8711418
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8731462, upper bound: 0.8711425
IS_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8708995, upper bound: 0.8726874
IS_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8728255, upper bound: 0.8726871
IS_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8712182, upper bound: 0.8731469
IS_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.60
Output dim: 0, lower bound: -0.8731462, upper bound: 0.8731481

## BFS IS instance: IS_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 6.5982318, 9.0338192, 6.6573234, 9.0247841, -1.8335357, 1.8002970
1: -17.5370369, -13.7700310, -17.4725952, -13.8015013, -2.6780024, 2.6434336
2: -3.2814131, -0.4972646, -3.2599447, -0.5209533, -2.3620605, 2.3634613
3: -10.8991728, -7.9380732, -10.8533945, -7.9794688, -2.4733934, 2.4624462
4: -12.5791636, -8.9994106, -12.5936823, -9.0311022, -2.7593989, 2.7922060
5: -4.9702191, -2.5876827, -4.9064183, -2.6704903, -1.8804531, 1.8696442
6: -3.0931189, -0.4395986, -2.9967294, -0.5493813, -2.1300600, 2.1014538
7: -9.4177418, -5.3944197, -9.4092741, -5.4664798, -2.9911633, 3.0238302
8: -2.6259117, -0.3371201, -2.5912337, -0.3282342, -2.0490079, 2.0306354
9: -4.5053473, -1.7431841, -4.4986444, -1.7705318, -2.2890568, 2.2910836

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 5773
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_B1_B2_A2_B1_A1

### Relational analysis result of IS_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8664622, upper bound: 0.8726856
time: 5.25 seconds

## Relational analysis of IS_B1_B2_A2_B1_A2

### Relational analysis result of IS_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683891, upper bound: 0.8726875
time: 11.98 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 6.6455879, 9.0276957, 6.5993776, 9.0330334, -1.8317847, 1.8705226
1: -17.4770412, -13.7815952, -17.5287514, -13.7714787, -2.7660465, 2.8169365
2: -3.2672682, -0.5233757, -3.2783585, -0.4975679, -2.3625703, 2.3438618
3: -10.8627720, -7.9575853, -10.9030266, -7.9387708, -2.5943427, 2.6235909
4: -12.5311928, -9.0332766, -12.5771513, -9.0181723, -2.7580094, 2.7817354
5: -4.9585772, -2.6730518, -4.9641061, -2.5860720, -2.0767064, 2.0349820
6: -3.0695581, -0.5577631, -3.0846047, -0.4368048, -2.3020239, 2.2697031
7: -9.3273392, -5.4201288, -9.4205809, -5.4277616, -3.1332912, 3.1957221
8: -2.5915947, -0.3481894, -2.6209645, -0.3384337, -2.0337591, 2.0523167
9: -4.4631495, -1.7603885, -4.5038137, -1.7579497, -2.2783504, 2.3061194

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726739, upper bound: 0.8706740
time: 8.07 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726739, upper bound: 0.8706702
time: 5.41 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 6.6243072, 9.0346498, 6.5967450, 9.0346003, -1.8548167, 1.8790261
1: -17.5260201, -13.7546463, -17.5305176, -13.7630653, -2.8244448, 2.8408184
2: -3.2837861, -0.5076383, -3.2812831, -0.4936457, -2.3800156, 2.3641174
3: -10.9037743, -7.9280925, -10.9046364, -7.9296498, -2.6455479, 2.6518922
4: -12.5474520, -9.0141430, -12.5802574, -9.0147133, -2.7874823, 2.8071678
5: -4.9836311, -2.6585593, -4.9658065, -2.5818915, -2.0935640, 2.0509775
6: -3.0829544, -0.5421209, -3.0887029, -0.4356294, -2.3181496, 2.2907186
7: -9.3506374, -5.3705797, -9.4276152, -5.4237156, -3.1574216, 3.2232494
8: -2.6071768, -0.3371859, -2.6234169, -0.3358188, -2.0519934, 2.0657334
9: -4.4939737, -1.7317178, -4.5114098, -1.7562912, -2.3068943, 2.3222442

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_B2_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8711444, upper bound: 0.8711417
time: 5.79 seconds

## Relational analysis of IS_B2_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8711444, upper bound: 0.8711426
time: 10.47 seconds

## BFS IS instance: IS_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 6.6508741, 9.0239611, 6.5980577, 9.0455780, -1.8422499, 1.8664182
1: -17.4756069, -13.7780275, -17.5348568, -13.7644444, -2.7819390, 2.8197899
2: -3.2671645, -0.5290669, -3.2822876, -0.4926099, -2.3623357, 2.3452122
3: -10.8617859, -7.9558029, -10.9047813, -7.9369760, -2.5948577, 2.6219335
4: -12.5217190, -9.0240946, -12.6395092, -8.9943008, -2.7617278, 2.8043904
5: -4.9596148, -2.6713576, -4.9692259, -2.5793104, -2.0738358, 2.0387850
6: -3.0664613, -0.5618219, -3.0848966, -0.4178658, -2.3068461, 2.2630897
7: -9.3256187, -5.4095416, -9.5111942, -5.3992352, -3.1445928, 3.2175858
8: -2.5854836, -0.3489943, -2.6230555, -0.3143387, -2.0434766, 2.0506449
9: -4.4617839, -1.7545618, -4.5392246, -1.7390484, -2.2912865, 2.3081827

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8707499, upper bound: 0.8726743
time: 18.54 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8707469, upper bound: 0.8726866
time: 8.11 seconds

## BFS IS instance: IS_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 6.6401653, 9.0297699, 6.5929055, 9.0466766, -1.8508453, 1.8769419
1: -17.4833546, -13.7729750, -17.5381870, -13.7623625, -2.7815909, 2.8324547
2: -3.2729335, -0.5181476, -3.2840796, -0.4872651, -2.3740144, 2.3533626
3: -10.8661699, -7.9472170, -10.9068365, -7.9329429, -2.6071620, 2.6362219
4: -12.5356913, -9.0189018, -12.6461973, -8.9930458, -2.7768841, 2.8159173
5: -4.9636350, -2.6680422, -4.9709959, -2.5785036, -2.0819354, 2.0454392
6: -3.0785036, -0.5557728, -3.0912018, -0.4166865, -2.3155947, 2.2764156
7: -9.3364048, -5.3998303, -9.5139294, -5.3940239, -3.1610308, 3.2281466
8: -2.5994163, -0.3444967, -2.6305180, -0.3136735, -2.0504484, 2.0609968
9: -4.4724913, -1.7498060, -4.5447431, -1.7383757, -2.2998981, 2.3189137

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 453
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726751, upper bound: 0.8726767
time: 15.44 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726751, upper bound: 0.8726741
time: 5.60 seconds

## BFS IS instance: IS_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 6.6482472, 9.0255175, 6.5770006, 9.0525780, -1.8508224, 1.8850182
1: -17.4773788, -13.7696095, -17.5836716, -13.7374840, -2.8058586, 2.8576684
2: -3.2701252, -0.5251570, -3.2988544, -0.4768908, -2.3829479, 2.3626723
3: -10.8633986, -7.9467444, -10.9457312, -7.9073343, -2.6233292, 2.6633325
4: -12.5247984, -9.0206299, -12.6558647, -8.9751778, -2.7871761, 2.8265848
5: -4.9613428, -2.6671727, -4.9941502, -2.5648036, -2.0899734, 2.0704029
6: -3.0705636, -0.5606451, -3.0983086, -0.4022741, -2.3211999, 2.2791898
7: -9.3326521, -5.4054928, -9.5345182, -5.3494902, -3.2011833, 3.2416301
8: -2.5879307, -0.3463802, -2.6384945, -0.3033018, -2.0571012, 2.0678720
9: -4.4693537, -1.7529032, -4.5699983, -1.7103860, -2.3092184, 2.3366103

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_B2_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8692113, upper bound: 0.8731470
time: 7.41 seconds

## Relational analysis of IS_B2_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8692113, upper bound: 0.8711427
time: 13.82 seconds

## BFS IS instance: IS_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 6.6375360, 9.0313292, 6.5718503, 9.0536900, -1.8594303, 1.8955432
1: -17.4851265, -13.7645588, -17.5870018, -13.7354002, -2.8055162, 2.8658123
2: -3.2758875, -0.5142438, -3.3006537, -0.4715458, -2.3945456, 2.3708110
3: -10.8677845, -7.9381504, -10.9477797, -7.9032907, -2.6356487, 2.6754150
4: -12.5387745, -9.0154400, -12.6625681, -8.9739218, -2.8023186, 2.8381257
5: -4.9653573, -2.6638608, -4.9959164, -2.5640011, -2.0980783, 2.0770581
6: -3.0826035, -0.5545983, -3.1046093, -0.4011021, -2.3299556, 2.2925160
7: -9.3434372, -5.3957801, -9.5372562, -5.3442822, -3.2176151, 3.2522023
8: -2.6018629, -0.3418813, -2.6459360, -0.3026347, -2.0640907, 2.0782402
9: -4.4800773, -1.7481472, -4.5755520, -1.7097174, -2.3178544, 2.3473721

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5778
type: A, layer: 1, pos: 453
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 471

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5773

## Relational analysis of IS_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_B2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8711421, upper bound: 0.8731465
time: 8.45 seconds

## Relational analysis of IS_B2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8711421, upper bound: 0.8711429
time: 11.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 40.61 seconds
IS_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8664622, upper bound: 0.8726856
IS_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8683891, upper bound: 0.8726875
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8726739, upper bound: 0.8706740
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8726739, upper bound: 0.8706702
IS_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8711444, upper bound: 0.8711417
IS_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8711444, upper bound: 0.8711426
IS_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8707499, upper bound: 0.8726743
IS_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8707469, upper bound: 0.8726866
IS_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8726751, upper bound: 0.8726767
IS_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8726751, upper bound: 0.8726741
IS_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8692113, upper bound: 0.8731470
IS_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8692113, upper bound: 0.8711427
IS_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8711421, upper bound: 0.8731465
IS_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 40.61
Output dim: 0, lower bound: -0.8711421, upper bound: 0.8711429
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8442518711090088
rel_dist={0: [-0.8742898229747036, 0.874290832632278]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2438.04 seconds
