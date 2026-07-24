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
execution time: IAR + LP analysis = 14.63 + 33.64 = 48.27 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.73 seconds, max iter: 100)

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
rel_dist={0: [-0.6370174797469685, 0.637020759518645]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7830688953399658
rel_dist={0: [-0.7572621253385305, 0.757258685054949]}

## Binary Search Result
Binary search time: 219.84 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3331.89 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2852455
time: 5.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2945697
time: 5.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.68
Output dim: 0, lower bound: -1.2945698, upper bound: 1.2852455
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.68
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

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852456
time: 5.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852459
time: 5.48 seconds

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

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2945695
time: 5.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2945696
time: 5.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.74 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.74
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852456
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.74
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2852459
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.74
Output dim: 0, lower bound: -1.2852461, upper bound: 1.2945695
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.74
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

Time for backsubstitution: 12.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809106, upper bound: 1.2852392
time: 8.14 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852423, upper bound: 1.2852389
time: 5.57 seconds

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
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852398
time: 6.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852422, upper bound: 1.2852392
time: 5.71 seconds

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

Time for backsubstitution: 13.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639
time: 8.05 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852390, upper bound: 1.2945637
time: 5.90 seconds

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

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945652
time: 6.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852391, upper bound: 1.2945646
time: 6.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2809106, upper bound: 1.2852392
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2852423, upper bound: 1.2852389
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852398
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2852422, upper bound: 1.2852392
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945639
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2852390, upper bound: 1.2945637
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945652
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.70
Output dim: 0, lower bound: -1.2852391, upper bound: 1.2945646

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

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2809080
time: 7.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852422
time: 9.06 seconds

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

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829723, upper bound: 1.2852393
time: 13.86 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852393, upper bound: 1.2852393
time: 5.31 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.5904884, 9.0354805, -2.0454321, 2.1149426
1: -17.4648304, -13.8021736, -17.5395622, -13.7612896, -3.0432825, 3.1981316
2: -3.2573876, -0.5272779, -3.2866726, -0.4902350, -2.6562266, 2.6416388
3: -10.8511620, -7.9760609, -10.9079905, -7.9272232, -2.7615380, 2.9079161
4: -12.5276699, -9.0527420, -12.5829248, -8.9888010, -3.0916071, 3.0865169
5: -4.9013195, -2.6738970, -4.9724512, -2.5807762, -2.2134852, 2.1263869
6: -2.9942713, -0.5683670, -3.0983086, -0.4341698, -2.4129641, 2.3603847
7: -9.3230848, -5.4961648, -9.4304581, -5.3878284, -3.3604732, 3.4687977
8: -2.5841184, -0.3503575, -2.6340675, -0.3340945, -2.2198572, 2.2561331
9: -4.4652653, -1.7884412, -4.5135813, -1.7374017, -2.4934416, 2.4876738

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902586, upper bound: 1.2809049
time: 5.63 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902586, upper bound: 1.2852390
time: 5.32 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.5904913, 9.0354805, -2.0541754, 2.1188946
1: -17.4762993, -13.7852192, -17.5395584, -13.7612877, -3.0552826, 3.2152624
2: -3.2660034, -0.5133837, -3.2866712, -0.4902354, -2.6652799, 2.6561909
3: -10.8567801, -7.9611416, -10.9079885, -7.9272246, -2.7670860, 2.9256120
4: -12.6000013, -9.0257072, -12.5829220, -8.9888144, -3.1647029, 3.1138220
5: -4.9100409, -2.6617544, -4.9724493, -2.5807762, -2.2230120, 2.1368718
6: -3.0051947, -0.5469360, -3.0983055, -0.4341702, -2.4240465, 2.3728044
7: -9.4240265, -5.4593987, -9.4304562, -5.3878465, -3.4065251, 3.5068052
8: -2.5963993, -0.3227391, -2.6340632, -0.3340960, -2.2323809, 2.2609944
9: -4.5145378, -1.7673830, -4.5135813, -1.7374126, -2.5180178, 2.5091338

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920430, upper bound: 1.2852366
time: 11.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945601, upper bound: 1.2852369
time: 5.81 seconds

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
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902582
time: 7.04 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945642
time: 8.54 seconds

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

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945608
time: 14.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852361, upper bound: 1.2945600
time: 5.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.5904884, 9.0354805, -2.1291571, 2.1347342
1: -17.5305214, -13.7629414, -17.5395622, -13.7612896, -3.2279596, 3.2355165
2: -3.2812915, -0.4936304, -3.2866726, -0.4902350, -2.6677051, 2.6700132
3: -10.9046412, -7.9296236, -10.9079905, -7.9272232, -2.9562764, 2.9580698
4: -12.5802679, -9.0147047, -12.5829248, -8.9888010, -3.1530099, 3.1297121
5: -4.9658098, -2.5817404, -4.9724512, -2.5807762, -2.2797050, 2.2861130
6: -3.0887170, -0.4356251, -3.0983086, -0.4341698, -2.5024893, 2.5106957
7: -9.4276257, -5.4237018, -9.4304581, -5.3878284, -3.5758672, 3.5426450
8: -2.6234269, -0.3358107, -2.6340675, -0.3340945, -2.2627635, 2.2714975
9: -4.5114365, -1.7562879, -4.5135813, -1.7374017, -2.5353990, 2.5185695

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902585
time: 6.67 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945644
time: 5.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 6.5878325, 9.0499220, 6.5904913, 9.0354805, -2.1378248, 2.1399722
1: -17.5419140, -13.7459955, -17.5395584, -13.7612877, -3.2398920, 3.2526455
2: -3.2899241, -0.4797205, -3.2866712, -0.4902354, -2.6767435, 2.6845374
3: -10.9102325, -7.9146242, -10.9079885, -7.9272246, -2.9617558, 2.9758258
4: -12.6525669, -8.9876604, -12.5829220, -8.9888144, -3.2040377, 3.1570163
5: -4.9745350, -2.5696204, -4.9724493, -2.5807762, -2.2892628, 2.2892928
6: -3.0996435, -0.4142132, -3.0983055, -0.4341702, -2.5136416, 2.5166686
7: -9.5286961, -5.3869095, -9.4304562, -5.3878465, -3.5920105, 3.5809968
8: -2.6356602, -0.3081894, -2.6340632, -0.3340960, -2.2752886, 2.2763774
9: -4.5606918, -1.7352251, -4.5135813, -1.7374126, -2.5525656, 2.5399845

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945615
time: 5.77 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852361, upper bound: 1.2945606
time: 11.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.26 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2809080
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2809084, upper bound: 1.2852422
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2829723, upper bound: 1.2852393
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2852393, upper bound: 1.2852393
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2902586, upper bound: 1.2809049
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2902586, upper bound: 1.2852390
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2920430, upper bound: 1.2852366
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2945601, upper bound: 1.2852369
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902582
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945642
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945608
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2852361, upper bound: 1.2945600
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902585
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945644
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945615
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -1.2852361, upper bound: 1.2945606

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

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2786446
time: 6.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2809110
time: 7.59 seconds

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

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2829725
time: 8.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2852394
time: 7.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 6.6529236, 9.0273266, 6.6629682, 9.0052099, -2.0225158, 2.0362420
1: -17.4759140, -13.7861834, -17.4686508, -13.8133907, -2.9998684, 3.0204206
2: -3.2658522, -0.5140333, -3.2607179, -0.5323584, -2.6168699, 2.6302290
3: -10.8563633, -7.9613185, -10.8490028, -7.9760637, -2.7129898, 2.7232656
4: -12.5997467, -9.0261335, -12.5268726, -9.0325241, -3.1223249, 3.0549746
5: -4.9090657, -2.6619830, -4.8949733, -2.6760564, -2.0534396, 2.0558903
6: -3.0044756, -0.5472517, -2.9943218, -0.5712204, -2.2587695, 2.2741711
7: -9.4235935, -5.4606194, -9.3197746, -5.4766378, -3.3306847, 3.2810450
8: -2.5961275, -0.3234749, -2.5910425, -0.3584452, -2.2043872, 2.2219980
9: -4.5142183, -1.7680771, -4.4630542, -1.7788218, -2.4785295, 2.4567654

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829723, upper bound: 1.2829725
time: 6.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829723, upper bound: 1.2852395
time: 13.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 6.6523504, 9.0279493, 6.6328669, 9.0185108, -2.0359769, 2.0668421
1: -17.4762955, -13.7852268, -17.5180206, -13.7986603, -3.0150099, 3.0708823
2: -3.2660027, -0.5133882, -3.2837667, -0.5127327, -2.6488810, 2.6534452
3: -10.8567772, -7.9611416, -10.8609304, -7.9565716, -2.7340536, 2.7413819
4: -12.6000013, -9.0257111, -12.5586567, -9.0211554, -3.1336002, 3.0880980
5: -4.9100347, -2.6617551, -4.9187069, -2.6455321, -2.0847936, 2.0926361
6: -3.0051887, -0.5469379, -3.0189347, -0.5446544, -2.2867427, 2.3108697
7: -9.4240208, -5.4594064, -9.3571053, -5.4563465, -3.3528047, 3.3194690
8: -2.5963984, -0.3227453, -2.6155782, -0.3383126, -2.2237649, 2.2340479
9: -4.5145350, -1.7673872, -4.4903383, -1.7632895, -2.4980216, 2.4853525

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2829728
time: 6.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2852416
time: 7.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902550, upper bound: 1.2786417
time: 6.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902550, upper bound: 1.2809072
time: 8.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.5878325, 9.0499220, -2.0595527, 2.1171274
1: -17.4648304, -13.8021736, -17.5419140, -13.7459955, -3.0587997, 3.2008338
2: -3.2573876, -0.5272779, -3.2899241, -0.4797205, -2.6632230, 2.6453090
3: -10.8511620, -7.9760609, -10.9102325, -7.9146242, -2.7769899, 2.9096079
4: -12.5276699, -9.0527420, -12.6525669, -8.9876604, -3.0926757, 3.1317329
5: -4.9013195, -2.6738970, -4.9745350, -2.5696204, -2.2155910, 2.1286092
6: -2.9942713, -0.5683670, -3.0996435, -0.4142132, -2.4175625, 2.3618469
7: -9.3230848, -5.4961648, -9.5286961, -5.3869095, -3.3624353, 3.4821630
8: -2.5841184, -0.3503575, -2.6356602, -0.3081894, -2.2295580, 2.2577934
9: -4.4652653, -1.7884412, -4.5606918, -1.7352251, -2.4958606, 2.4962409

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902550, upper bound: 1.2829701
time: 5.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2902549, upper bound: 1.2852363
time: 8.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 6.6529236, 9.0273266, 6.5984488, 9.0270367, -2.0448852, 2.1108074
1: -17.4759140, -13.7861834, -17.5342293, -13.7741671, -3.0418868, 3.2069120
2: -3.2658522, -0.5140333, -3.2840753, -0.4986792, -2.6517744, 2.6483459
3: -10.8563633, -7.9613185, -10.9006910, -7.9297953, -2.7614655, 2.9139843
4: -12.5997467, -9.0261335, -12.5793333, -8.9945755, -3.1588306, 3.1097264
5: -4.9090657, -2.6619830, -4.9593983, -2.5844667, -2.2173443, 2.1220264
6: -3.0044756, -0.5472517, -3.0886784, -0.4394197, -2.4170866, 2.3601406
7: -9.4235935, -5.4606194, -9.4230404, -5.4042864, -3.3890648, 3.4961073
8: -2.5961275, -0.3234749, -2.6302123, -0.3439407, -2.2203650, 2.2569292
9: -4.5142183, -1.7680771, -4.5090485, -1.7472069, -2.5054212, 2.5034266

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920425, upper bound: 1.2829690
time: 21.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920425, upper bound: 1.2852362
time: 7.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 6.6523504, 9.0279493, 6.5679331, 9.0403290, -2.0583615, 2.1329362
1: -17.4762955, -13.7852268, -17.5837746, -13.7594986, -3.0569425, 3.2418134
2: -3.2660027, -0.5133882, -3.3078647, -0.4789677, -2.6774397, 2.6708078
3: -10.8567772, -7.9611416, -10.9148149, -7.9100828, -2.7826452, 2.9430203
4: -12.6000013, -9.0257111, -12.6112347, -8.9841499, -3.1692724, 3.1429949
5: -4.9100347, -2.6617551, -4.9830999, -2.5534453, -2.2321038, 2.1529865
6: -3.0051887, -0.5469379, -3.1129391, -0.4123597, -2.4316196, 2.3917651
7: -9.4240208, -5.4594064, -9.4612961, -5.3838530, -3.4108343, 3.5206881
8: -2.5963984, -0.3227453, -2.6548886, -0.3238478, -2.2399125, 2.2690978
9: -4.5145350, -1.7673872, -4.5365224, -1.7315118, -2.5250726, 2.5276308

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945602, upper bound: 1.2829697
time: 5.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945602, upper bound: 1.2852368
time: 6.05 seconds

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

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2877438
time: 9.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2902597
time: 7.83 seconds

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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2920429
time: 10.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2945601
time: 9.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 6.5884185, 9.0492897, 6.6629682, 9.0052099, -2.1084321, 2.0587223
1: -17.5415192, -13.7469587, -17.4686508, -13.8133907, -3.1865501, 3.0624518
2: -3.2897334, -0.4803684, -3.2607179, -0.5323584, -2.6351933, 2.6606777
3: -10.9096813, -7.9148140, -10.8490028, -7.9760637, -2.9024301, 2.7719111
4: -12.6522999, -8.9880905, -12.5268726, -9.0325241, -3.1519198, 3.0915689
5: -4.9735570, -2.5699005, -4.8949733, -2.6760564, -2.1218853, 2.2096727
6: -3.0989206, -0.4145989, -2.9943218, -0.5712204, -2.3548274, 2.4164662
7: -9.5281649, -5.3881407, -9.3197746, -5.4766378, -3.4974852, 3.3577075
8: -2.6353793, -0.3089285, -2.5910425, -0.3584452, -2.2482400, 2.2365363
9: -4.5603604, -1.7359605, -4.4630542, -1.7788218, -2.5029068, 2.4918897

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2920446
time: 9.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945600
time: 13.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 6.5878396, 9.0499163, 6.6328669, 9.0185108, -2.1219840, 2.0852122
1: -17.5419121, -13.7460022, -17.5180206, -13.7986603, -3.2020969, 3.1129146
2: -3.2899220, -0.4797248, -3.2837667, -0.5127327, -2.6661525, 2.6778874
3: -10.9102278, -7.9146247, -10.8609304, -7.9565716, -2.9232731, 2.7900336
4: -12.6525650, -8.9876623, -12.5586567, -9.0211554, -3.1631970, 3.1246982
5: -4.9745283, -2.5696230, -4.9187069, -2.6455321, -2.1409655, 2.2343051
6: -3.0996380, -0.4142151, -3.0189347, -0.5446544, -2.3719342, 2.4420407
7: -9.5286942, -5.3869162, -9.3571053, -5.4563465, -3.5202575, 3.3961377
8: -2.6356573, -0.3081956, -2.6155782, -0.3383126, -2.2676516, 2.2486136
9: -4.5606894, -1.7352314, -4.4903383, -1.7632895, -2.5224042, 2.5204821

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852384, upper bound: 1.2920426
time: 20.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852362, upper bound: 1.2945611
time: 8.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 42.61 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2786446
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2809110
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2829725
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809056, upper bound: 1.2852394
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2829723, upper bound: 1.2829725
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2829723, upper bound: 1.2852395
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2829728
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2852394, upper bound: 1.2852416
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2902550, upper bound: 1.2786417
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2902550, upper bound: 1.2809072
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2902550, upper bound: 1.2829701
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2902549, upper bound: 1.2852363
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2920425, upper bound: 1.2829690
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2920425, upper bound: 1.2852362
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2945602, upper bound: 1.2829697
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2945602, upper bound: 1.2852368
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2877438
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2902597
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2920429
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2809024, upper bound: 1.2945601
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2920446
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945600
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2852384, upper bound: 1.2920426
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 0, lower bound: -1.2852362, upper bound: 1.2945611
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2902585
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 0, lower bound: -1.2809052, upper bound: 1.2945644
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 0, lower bound: -1.2829691, upper bound: 1.2945615
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 0, lower bound: -1.2852361, upper bound: 1.2945606
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.088984727859497
rel_dist={0: [-1.2945795095372876, 1.2945797204952658]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9778986
time: 15.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9840132
time: 6.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 22.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 22.65
Output dim: 0, lower bound: -0.9840133, upper bound: 0.9778986
IS_A2, status: Status.UNKNOWN, split count: 1, time: 22.65
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

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9778987
time: 7.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9778984
time: 5.57 seconds

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

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9840132
time: 6.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9840130
time: 5.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.33
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9778987
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.33
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9778984
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.33
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9840132
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.33
Output dim: 0, lower bound: -0.9778986, upper bound: 0.9840130

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.6551013, 9.0135641, -1.8451233, 1.8451233
1: -17.4738808, -13.8005133, -17.4738808, -13.8005133, -2.7142854, 2.7142849
2: -3.2627931, -0.5239117, -3.2627931, -0.5239117, -2.4109025, 2.4109030
3: -10.8545074, -7.9736691, -10.8545074, -7.9736691, -2.5000105, 2.5000103
4: -12.5303125, -9.0268316, -12.5303125, -9.0268316, -2.8096266, 2.8096275
5: -4.9079676, -2.6729443, -4.9079676, -2.6729443, -1.8929682, 1.8929679
6: -3.0038805, -0.5669265, -3.0038805, -0.5669265, -2.1156428, 2.1156425
7: -9.3258791, -5.4602871, -9.3258791, -5.4602871, -3.0392780, 3.0392780
8: -2.5947652, -0.3486171, -2.5947652, -0.3486171, -2.0738950, 2.0738947
9: -4.4674053, -1.7695539, -4.4674053, -1.7695539, -2.3120961, 2.3120961

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752814, upper bound: 0.9778959
time: 13.46 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779489, upper bound: 0.9778951
time: 8.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.5927343, 9.0354567, -1.8675480, 1.9272377
1: -17.4738808, -13.8005133, -17.5390453, -13.7614079, -2.7554607, 2.9087095
2: -3.2627931, -0.5239117, -3.2865353, -0.4918160, -2.4442258, 2.4282475
3: -10.8545074, -7.9736691, -10.9076443, -7.9280491, -2.5476837, 2.6835394
4: -12.5303125, -9.0268316, -12.5827179, -8.9927177, -2.8426619, 2.8646383
5: -4.9079676, -2.6729443, -4.9722934, -2.5815196, -2.0805249, 1.9520245
6: -3.0038805, -0.5669265, -3.0977759, -0.4350057, -2.2873373, 2.1915007
7: -9.3258791, -5.4602871, -9.4298363, -5.3892508, -3.1142607, 3.2548721
8: -2.5947652, -0.3486171, -2.6312370, -0.3341718, -2.0898914, 2.1125300
9: -4.4674053, -1.7695539, -4.5134907, -1.7386889, -2.3451862, 2.3569572

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752814, upper bound: 0.9778954
time: 5.89 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779489, upper bound: 0.9778954
time: 5.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 6.5927343, 9.0354567, 6.6551013, 9.0135641, -1.9272377, 1.8675478
1: -17.5390453, -13.7614079, -17.4738808, -13.8005133, -2.9087090, 2.7554607
2: -3.2865353, -0.4918160, -3.2627931, -0.5239117, -2.4282475, 2.4442258
3: -10.9076443, -7.9280491, -10.8545074, -7.9736691, -2.6835399, 2.5476835
4: -12.5827179, -8.9927177, -12.5303125, -9.0268316, -2.8646388, 2.8426619
5: -4.9722934, -2.5815196, -4.9079676, -2.6729443, -1.9520245, 2.0805249
6: -3.0977759, -0.4350057, -3.0038805, -0.5669265, -2.1915007, 2.2873373
7: -9.4298363, -5.3892508, -9.3258791, -5.4602871, -3.2548723, 3.1142607
8: -2.6312370, -0.3341718, -2.5947652, -0.3486171, -2.1125298, 2.0898910
9: -4.5134907, -1.7386889, -4.4674053, -1.7695539, -2.3569567, 2.3451862

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752278, upper bound: 0.9840090
time: 7.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778953, upper bound: 0.9840101
time: 9.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.5904884, 9.0354805, -1.9490502, 1.9490499
1: -17.5395622, -13.7612896, -17.5395622, -13.7612896, -2.9458828, 2.9458823
2: -3.2866726, -0.4902350, -3.2866726, -0.4902350, -2.4550834, 2.4550831
3: -10.9079905, -7.9272232, -10.9079905, -7.9272232, -2.7340951, 2.7340941
4: -12.5829248, -8.9888010, -12.5829248, -8.9888010, -2.9025865, 2.9025869
5: -4.9724512, -2.5807762, -4.9724512, -2.5807762, -2.1463261, 2.1463258
6: -3.0983086, -0.4341698, -3.0983086, -0.4341698, -2.3729398, 2.3729398
7: -9.4304581, -5.3878284, -9.4304581, -5.3878284, -3.3268304, 3.3268304
8: -2.6340675, -0.3340945, -2.6340675, -0.3340945, -2.1299858, 2.1299858
9: -4.5135813, -1.7374017, -4.5135813, -1.7374017, -2.3825750, 2.3825753

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752278, upper bound: 0.9840099
time: 5.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778953, upper bound: 0.9840098
time: 5.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9752814, upper bound: 0.9778959
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9779489, upper bound: 0.9778951
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9752814, upper bound: 0.9778954
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9779489, upper bound: 0.9778954
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9752278, upper bound: 0.9840090
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9778953, upper bound: 0.9840101
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9752278, upper bound: 0.9840099
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.55
Output dim: 0, lower bound: -0.9778953, upper bound: 0.9840098

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.6572323, 9.0132694, -1.8382807, 1.8419440
1: -17.4648304, -13.8021736, -17.4707909, -13.8010731, -2.7045574, 2.7095361
2: -3.2573876, -0.5272779, -3.2609453, -0.5250452, -2.4044504, 2.4059339
3: -10.8511620, -7.9760609, -10.8533611, -7.9744825, -2.4955606, 2.4966741
4: -12.5276699, -9.0527420, -12.5294170, -9.0356874, -2.7980709, 2.7827082
5: -4.9013195, -2.6738970, -4.9056940, -2.6732683, -1.8850732, 1.8893876
6: -2.9942713, -0.5683670, -3.0005960, -0.5674138, -2.1054244, 2.1109505
7: -9.3230848, -5.4961648, -9.3249321, -5.4725504, -3.0242019, 3.0023346
8: -2.5841184, -0.3503575, -2.5911255, -0.3492079, -2.0624461, 2.0682046
9: -4.4652653, -1.7884412, -4.4666796, -1.7760175, -2.3035498, 2.2925100

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9739100, upper bound: 0.9779473
time: 22.18 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752795, upper bound: 0.9779472
time: 8.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.6551056, 9.0135641, -1.8459449, 1.8592029
1: -17.4762993, -13.7852192, -17.4738712, -13.8005123, -2.7147136, 2.7298017
2: -3.2660034, -0.5133837, -3.2627916, -0.5239134, -2.4140911, 2.4222851
3: -10.8567801, -7.9611416, -10.8545055, -7.9736719, -2.5013885, 2.5153921
4: -12.6000013, -9.0257072, -12.5303106, -9.0268536, -2.8654819, 2.8011773
5: -4.9100409, -2.6617544, -4.9079614, -2.6729436, -1.8937407, 1.9044530
6: -3.0051947, -0.5469360, -3.0038753, -0.5669270, -2.1151879, 2.1363673
7: -9.4240265, -5.4593987, -9.3258781, -5.4603238, -3.0922184, 3.0292125
8: -2.5963993, -0.3227391, -2.5947590, -0.3486185, -2.0716000, 2.0832930
9: -4.5145378, -1.7673830, -4.4674058, -1.7695721, -2.3395715, 2.3084636

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765775, upper bound: 0.9779474
time: 6.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9779470, upper bound: 0.9779467
time: 7.58 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 6.6613379, 9.0126896, 6.5948710, 9.0351601, -1.8607047, 1.9236597
1: -17.4648304, -13.8021736, -17.5359573, -13.7619677, -2.7457337, 2.9039602
2: -3.2573876, -0.5272779, -3.2846940, -0.4929603, -2.4377632, 2.4232748
3: -10.8511620, -7.9760609, -10.9064960, -7.9288654, -2.5432324, 2.6802592
4: -12.5276699, -9.0527420, -12.5818176, -9.0015697, -2.8311062, 2.8377037
5: -4.9013195, -2.6738970, -4.9700222, -2.5818481, -2.0726910, 1.9480028
6: -2.9942713, -0.5683670, -3.0944965, -0.4354992, -2.2772851, 2.1862376
7: -9.3230848, -5.4961648, -9.4288807, -5.4015121, -3.0991869, 3.2179210
8: -2.5841184, -0.3503575, -2.6276016, -0.3347530, -2.0784469, 2.1060557
9: -4.4652653, -1.7884412, -4.5127640, -1.7451525, -2.3366690, 2.3373563

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9800248, upper bound: 0.9778938
time: 8.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9813377, upper bound: 0.9778939
time: 10.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 6.6523466, 9.0279541, 6.5927405, 9.0354567, -1.8683701, 1.9302198
1: -17.4762993, -13.7852192, -17.5390377, -13.7614098, -2.7558889, 2.9242253
2: -3.2660034, -0.5133837, -3.2865326, -0.4918187, -2.4474144, 2.4396768
3: -10.8567801, -7.9611416, -10.9076405, -7.9280491, -2.5490632, 2.6992474
4: -12.6000013, -9.0257072, -12.5827150, -8.9927397, -2.8959842, 2.8561873
5: -4.9100409, -2.6617544, -4.9722881, -2.5815203, -2.0811119, 1.9543138
6: -3.0051947, -0.5469360, -3.0977705, -0.4350076, -2.2869420, 2.1964145
7: -9.4240265, -5.4593987, -9.4298363, -5.3892851, -3.1338387, 3.2448692
8: -2.5963993, -0.3227391, -2.6312332, -0.3341732, -2.0875978, 2.1153603
9: -4.5145378, -1.7673830, -4.5134888, -1.7387085, -2.3605070, 2.3533449

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9826967, upper bound: 0.9778963
time: 8.88 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840072, upper bound: 0.9778933
time: 8.40 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 6.5989819, 9.0345812, 6.6572323, 9.0132694, -1.9203982, 1.8643661
1: -17.5300007, -13.7630663, -17.4707909, -13.8010731, -2.8989806, 2.7507114
2: -3.2811546, -0.4952123, -3.2609453, -0.5250452, -2.4217858, 2.4392321
3: -10.9042959, -7.9304504, -10.8533611, -7.9744825, -2.6790857, 2.5443482
4: -12.5800629, -9.0186205, -12.5294170, -9.0356874, -2.8530231, 2.8157477
5: -4.9656525, -2.5824852, -4.9056940, -2.6732683, -1.9440267, 2.0765207
6: -3.0881839, -0.4364614, -3.0005960, -0.5674138, -2.1812704, 2.2820938
7: -9.4270048, -5.4251251, -9.3249321, -5.4725504, -3.2370319, 3.0773244
8: -2.6205978, -0.3358879, -2.5911255, -0.3492079, -2.1010575, 2.0842133
9: -4.5113444, -1.7575762, -4.4666796, -1.7760175, -2.3470025, 2.3256104

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9738564, upper bound: 0.9840068
time: 9.78 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752260, upper bound: 0.9840069
time: 5.44 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 6.5900803, 9.0499001, 6.6551056, 9.0135641, -1.9280567, 1.8802048
1: -17.5413933, -13.7461166, -17.4738712, -13.8005123, -2.9090681, 2.7709737
2: -3.2897859, -0.4813030, -3.2627916, -0.5239134, -2.4314647, 2.4501808
3: -10.9098873, -7.9154501, -10.8545055, -7.9736719, -2.6848249, 2.5631349
4: -12.6523581, -8.9915752, -12.5303106, -9.0268536, -2.8938465, 2.8342252
5: -4.9743786, -2.5703549, -4.9079614, -2.6729436, -1.9527347, 2.0826285
6: -3.0991123, -0.4150491, -3.0038753, -0.5669270, -2.1910479, 2.2919338
7: -9.5280743, -5.3883324, -9.3258781, -5.4603238, -3.2682357, 3.1042132
8: -2.6328292, -0.3082666, -2.5947590, -0.3486185, -2.1102357, 2.0969219
9: -4.5606008, -1.7365143, -4.4674058, -1.7695721, -2.3638630, 2.3415575

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765238, upper bound: 0.9840078
time: 12.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778934, upper bound: 0.9840070
time: 23.82 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.5926256, 9.0351839, -1.9422097, 1.9454694
1: -17.5305214, -13.7629414, -17.5364723, -13.7618465, -2.9361506, 2.9411321
2: -3.2812915, -0.4936304, -3.2848313, -0.4913787, -2.4485674, 2.4500701
3: -10.9046412, -7.9296236, -10.9068413, -7.9280386, -2.7296405, 2.7308125
4: -12.5802679, -9.0147047, -12.5820246, -8.9976549, -2.8909683, 2.8756418
5: -4.9658098, -2.5817404, -4.9701805, -2.5811043, -2.1384773, 2.1423185
6: -3.0887170, -0.4356251, -3.0950296, -0.4346628, -2.3628917, 2.3676965
7: -9.4276257, -5.4237018, -9.4294987, -5.4000888, -3.3089962, 3.2898884
8: -2.6234269, -0.3358107, -2.6304307, -0.3346772, -2.1185169, 2.1235220
9: -4.5114365, -1.7562879, -4.5128541, -1.7438633, -2.3740807, 2.3630085

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9738563, upper bound: 0.9840078
time: 8.23 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9752259, upper bound: 0.9840080
time: 5.47 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 6.5878325, 9.0499220, 6.5904932, 9.0354805, -1.9498682, 1.9520714
1: -17.5419140, -13.7459955, -17.5395584, -13.7612877, -2.9462399, 2.9613929
2: -3.2899241, -0.4797205, -3.2866697, -0.4902364, -2.4582038, 2.4660375
3: -10.9102325, -7.9146242, -10.9079876, -7.9272237, -2.7353797, 2.7498636
4: -12.6525669, -8.9876604, -12.5829220, -8.9888229, -2.9348516, 2.8941259
5: -4.9745350, -2.5696204, -4.9724469, -2.5807767, -2.1469297, 2.1484292
6: -3.0996435, -0.4142132, -3.0983026, -0.4341702, -2.3725653, 2.3775375
7: -9.5286961, -5.3869095, -9.4304581, -5.3878617, -3.3401957, 3.3168452
8: -2.6356602, -0.3081894, -2.6340623, -0.3340969, -2.1276913, 2.1328444
9: -4.5606918, -1.7352251, -4.5135808, -1.7374201, -2.3941643, 2.3789010

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765237, upper bound: 0.9840070
time: 5.45 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778933, upper bound: 0.9840070
time: 5.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.93 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9739100, upper bound: 0.9779473
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9752795, upper bound: 0.9779472
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9765775, upper bound: 0.9779474
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9779470, upper bound: 0.9779467
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9800248, upper bound: 0.9778938
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9813377, upper bound: 0.9778939
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9826967, upper bound: 0.9778963
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9840072, upper bound: 0.9778933
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9738564, upper bound: 0.9840068
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9752260, upper bound: 0.9840069
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9765238, upper bound: 0.9840078
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9778934, upper bound: 0.9840070
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9738563, upper bound: 0.9840078
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9752259, upper bound: 0.9840080
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9765237, upper bound: 0.9840070
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 0, lower bound: -0.9778933, upper bound: 0.9840070

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 6.6643982, 9.0094013, 6.6650982, 9.0049143, -1.8269577, 1.8318243
1: -17.4627972, -13.8072300, -17.4655590, -13.8139524, -2.6897521, 2.6998754
2: -3.2565873, -0.5306487, -3.2588725, -0.5334889, -2.3887343, 2.3939364
3: -10.8489895, -7.9769950, -10.8478575, -7.9768734, -2.4873729, 2.4862795
4: -12.5263281, -9.0549755, -12.5259790, -9.0413666, -2.7911444, 2.7769413
5: -4.8962097, -2.6751018, -4.8927040, -2.6763773, -1.8728776, 1.8727198
6: -2.9905038, -0.5700345, -2.9910395, -0.5717072, -2.0933208, 2.0963061
7: -9.3207808, -5.5025826, -9.3188305, -5.4888806, -3.0048003, 2.9891634
8: -2.5826812, -0.3542223, -2.5874071, -0.3590355, -2.0497127, 2.0609732
9: -4.4635868, -1.7920790, -4.4623322, -1.7852679, -2.2898450, 2.2832441

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9704087, upper bound: 0.9771070
time: 5.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9730696, upper bound: 0.9771071
time: 5.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 6.6613445, 9.0126820, 6.6350002, 9.0182123, -1.8423934, 1.8666489
1: -17.4648247, -13.8021851, -17.5149364, -13.7992182, -2.7056837, 2.7544794
2: -3.2573857, -0.5272837, -3.2819118, -0.5138688, -2.4199872, 2.4212451
3: -10.8511562, -7.9760628, -10.8597860, -7.9573917, -2.5110946, 2.5041113
4: -12.5276651, -9.0527449, -12.5577631, -9.0299988, -2.8026042, 2.8118300
5: -4.9013090, -2.6738982, -4.9164386, -2.6458559, -1.9086633, 1.9068360
6: -2.9942632, -0.5683703, -3.0156546, -0.5451427, -2.1249926, 2.1315064
7: -9.3230801, -5.4961796, -9.3561611, -5.4685874, -3.0276051, 3.0329702
8: -2.5841155, -0.3503647, -2.6119423, -0.3388972, -2.0692930, 2.0822818
9: -4.4652624, -1.7884493, -4.4896159, -1.7697393, -2.3099408, 2.3155570

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9717784, upper bound: 0.9771071
time: 5.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9744393, upper bound: 0.9771070
time: 5.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 6.6553974, 9.0246649, 6.6629725, 9.0052090, -1.8346272, 1.8490827
1: -17.4742699, -13.7902775, -17.4686451, -13.8133926, -2.6999083, 2.7201400
2: -3.2651997, -0.5167651, -3.2607160, -0.5323595, -2.3983808, 2.4102871
3: -10.8546076, -7.9620790, -10.8490028, -7.9760633, -2.4932117, 2.5049973
4: -12.5986605, -9.0279388, -12.5268736, -9.0325327, -2.8585067, 2.7954133
5: -4.9049315, -2.6629632, -4.8949709, -2.6760564, -1.8815427, 1.8877826
6: -3.0014291, -0.5486035, -2.9943213, -0.5712209, -2.1030817, 2.1217194
7: -9.4217157, -5.4658127, -9.3197727, -5.4766531, -3.0727944, 3.0160422
8: -2.5949621, -0.3266029, -2.5910401, -0.3584452, -2.0588665, 2.0753462
9: -4.5128517, -1.7710238, -4.4630556, -1.7788286, -2.3258324, 2.2991941

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9730766, upper bound: 0.9771069
time: 6.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9757375, upper bound: 0.9771070
time: 5.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 6.6523538, 9.0279455, 6.6328688, 9.0185108, -1.8500605, 1.8792531
1: -17.4762936, -13.7852325, -17.5180168, -13.7986603, -2.7158389, 2.7747436
2: -3.2660019, -0.5133911, -3.2837648, -0.5127336, -2.4296303, 2.4363945
3: -10.8567762, -7.9611425, -10.8609285, -7.9565706, -2.5169382, 2.5228302
4: -12.5999985, -9.0257130, -12.5586548, -9.0211639, -2.8699698, 2.8302941
5: -4.9100308, -2.6617565, -4.9187059, -2.6455312, -1.9173479, 1.9219096
6: -3.0051839, -0.5469394, -3.0189331, -0.5446548, -2.1347549, 2.1569304
7: -9.4240189, -5.4594107, -9.3571053, -5.4563608, -3.0957165, 3.0598483
8: -2.5963969, -0.3227477, -2.6155758, -0.3383126, -2.0784450, 2.0915887
9: -4.5145340, -1.7673898, -4.4903374, -1.7632966, -2.3458204, 2.3315165

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9744462, upper bound: 0.9771071
time: 6.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9771071, upper bound: 0.9771071
time: 6.47 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 6.6643982, 9.0094013, 6.6029987, 9.0267181, -1.8493013, 1.9120449
1: -17.4627972, -13.8072300, -17.5306053, -13.7748547, -2.7308331, 2.8906870
2: -3.2565873, -0.5306487, -3.2820950, -0.5015383, -2.4219322, 2.4113514
3: -10.8489895, -7.9769950, -10.8991814, -7.9314961, -2.5348663, 2.6659431
4: -12.5263281, -9.0549755, -12.5782156, -9.0076733, -2.8237915, 2.8318410
5: -4.8962097, -2.6751018, -4.9569602, -2.5855651, -2.0622339, 1.9307897
6: -2.9905038, -0.5700345, -3.0848246, -0.4407763, -2.2665439, 2.1709664
7: -9.3207808, -5.5025826, -9.4214411, -5.4180727, -3.0795426, 3.1997495
8: -2.5826812, -0.3542223, -2.6235247, -0.3446026, -2.0655994, 2.0976264
9: -4.4635868, -1.7920790, -4.5082297, -1.7550528, -2.3219328, 2.3273549

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765195, upper bound: 0.9770535
time: 7.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9791690, upper bound: 0.9770538
time: 9.15 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 6.6613445, 9.0126820, 6.5732627, 9.0399904, -1.8647327, 1.9367399
1: -17.4648247, -13.8021851, -17.5795002, -13.7602329, -2.7464638, 2.9383402
2: -3.2573857, -0.5272837, -3.3057046, -0.4822371, -2.4454679, 2.4398613
3: -10.8511562, -7.9760628, -10.9129152, -7.9120398, -2.5584311, 2.6938243
4: -12.5276651, -9.0527449, -12.6099138, -8.9981861, -2.8335633, 2.8666563
5: -4.9013090, -2.6738982, -4.9806108, -2.5555573, -2.0809474, 1.9591858
6: -2.9942632, -0.5683703, -3.1089108, -0.4148507, -2.2838752, 2.2010419
7: -9.3230801, -5.4961796, -9.4588985, -5.3979888, -3.1015515, 3.2296925
8: -2.5841155, -0.3503647, -2.6474047, -0.3245673, -2.0852804, 2.1133926
9: -4.4652624, -1.7884493, -4.5355926, -1.7396896, -2.3419433, 2.3540313

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778380, upper bound: 0.9770533
time: 8.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9804905, upper bound: 0.9770541
time: 13.29 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 6.6553974, 9.0246649, 6.6008692, 9.0270138, -1.8569725, 1.9186065
1: -17.4742699, -13.7902775, -17.5336857, -13.7742977, -2.7409897, 2.9098723
2: -3.2651997, -0.5167651, -3.2839327, -0.5003970, -2.4315872, 2.4277160
3: -10.8546076, -7.9620790, -10.9003267, -7.9306817, -2.5406990, 2.6849341
4: -12.5986605, -9.0279388, -12.5791121, -8.9988384, -2.8886395, 2.8503242
5: -4.9049315, -2.6629632, -4.9592266, -2.5852399, -2.0706520, 1.9371026
6: -3.0014291, -0.5486035, -3.0880995, -0.4402857, -2.2762003, 2.1811423
7: -9.4217157, -5.4658127, -9.4223995, -5.4058452, -3.1142416, 3.2266989
8: -2.5949621, -0.3266029, -2.6271563, -0.3440213, -2.0747495, 2.1069336
9: -4.5128517, -1.7710238, -4.5089545, -1.7486140, -2.3458855, 2.3433390

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9791765, upper bound: 0.9770536
time: 8.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9818288, upper bound: 0.9770539
time: 12.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 6.6523538, 9.0279455, 6.5711246, 9.0402880, -1.8723993, 1.9432979
1: -17.4762936, -13.7852325, -17.5825787, -13.7596741, -2.7566209, 2.9455049
2: -3.2660019, -0.5133911, -3.3075557, -0.4810920, -2.4551415, 2.4509430
3: -10.8567762, -7.9611425, -10.9140606, -7.9112244, -2.5642662, 2.7128150
4: -12.5999985, -9.0257130, -12.6108141, -8.9893551, -2.8984923, 2.8851414
5: -4.9100308, -2.6617565, -4.9828753, -2.5552275, -2.0893636, 1.9655042
6: -3.0051839, -0.5469394, -3.1121860, -0.4143577, -2.2935300, 2.2112198
7: -9.4240189, -5.4594107, -9.4598560, -5.3857617, -3.1363435, 3.2566419
8: -2.5963969, -0.3227477, -2.6510358, -0.3239918, -2.0944271, 2.1226971
9: -4.5145340, -1.7673898, -4.5363212, -1.7332476, -2.3658392, 2.3700204

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9805013, upper bound: 0.9770536
time: 7.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9831537, upper bound: 0.9770544
time: 8.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 6.6025190, 9.0312500, 6.6650982, 9.0049143, -1.9080160, 1.8542075
1: -17.5276642, -13.7681513, -17.4655590, -13.8139524, -2.8804927, 2.7408886
2: -3.2800872, -0.4988667, -3.2588725, -0.5334889, -2.4062338, 2.4269617
3: -10.9012403, -7.9316015, -10.8478575, -7.9768734, -2.6661458, 2.5337577
4: -12.5785789, -9.0215721, -12.5259790, -9.0413666, -2.8459816, 2.8093262
5: -4.9604883, -2.5843301, -4.8927040, -2.6763773, -1.9304330, 2.0614092
6: -3.0842793, -0.4389596, -2.9910395, -0.5717072, -2.1677537, 2.2691491
7: -9.4238682, -5.4318361, -9.3188305, -5.4888806, -3.2131472, 3.0638156
8: -2.6186032, -0.3397880, -2.5874071, -0.3590355, -2.0880489, 2.0769069
9: -4.5095530, -1.7616583, -4.4623322, -1.7852679, -2.3331630, 2.3152332

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9703550, upper bound: 0.9831534
time: 5.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9730160, upper bound: 0.9831535
time: 5.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 6.5982771, 9.0345879, 6.6350002, 9.0182123, -1.9248762, 1.8876845
1: -17.5304794, -13.7630386, -17.5149364, -13.7992182, -2.8990436, 2.7958903
2: -3.2812676, -0.4947782, -3.2819118, -0.5138688, -2.4365830, 2.4495940
3: -10.9045963, -7.9302168, -10.8597860, -7.9573917, -2.6912460, 2.5520394
4: -12.5802107, -9.0175762, -12.5577631, -9.0299988, -2.8577456, 2.8458219
5: -4.9656901, -2.5817518, -4.9164386, -2.6458559, -1.9483857, 2.0861745
6: -3.0883493, -0.4356384, -3.0156546, -0.5451427, -2.1840718, 2.2970378
7: -9.4275780, -5.4247513, -9.3561611, -5.4685874, -3.2396717, 3.0951061
8: -2.6213870, -0.3358436, -2.6119423, -0.3388972, -2.1088991, 2.0960042
9: -4.5114198, -1.7572235, -4.4896159, -1.7697393, -2.3533342, 2.3484282

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9717247, upper bound: 0.9831536
time: 6.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9743857, upper bound: 0.9831536
time: 5.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 6.5936098, 9.0465717, 6.6629725, 9.0052090, -1.9156790, 1.8697038
1: -17.5390606, -13.7512016, -17.4686451, -13.8133926, -2.8905778, 2.7611499
2: -3.2887149, -0.4849662, -3.2607160, -0.5323595, -2.4159379, 2.4373684
3: -10.9068298, -7.9165950, -10.8490028, -7.9760633, -2.6718888, 2.5525410
4: -12.6508694, -8.9945278, -12.5268736, -9.0325327, -2.8866470, 2.8278055
5: -4.9692111, -2.5722232, -4.8949709, -2.6760564, -1.9391394, 2.0675116
6: -3.0952075, -0.4175491, -2.9943213, -0.5712209, -2.1775296, 2.2789936
7: -9.5249214, -5.3950410, -9.3197727, -5.4766531, -3.2443514, 3.0907049
8: -2.6308351, -0.3121657, -2.5910401, -0.3584452, -2.0972247, 2.0887935
9: -4.5588050, -1.7405984, -4.4630556, -1.7788286, -2.3500314, 2.3311870

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9730228, upper bound: 0.9831536
time: 5.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9756838, upper bound: 0.9831536
time: 7.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.19 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9704087, upper bound: 0.9771070
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9730696, upper bound: 0.9771071
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9717784, upper bound: 0.9771071
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9744393, upper bound: 0.9771070
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9730766, upper bound: 0.9771069
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9757375, upper bound: 0.9771070
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9744462, upper bound: 0.9771071
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9771071, upper bound: 0.9771071
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9765195, upper bound: 0.9770535
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9791690, upper bound: 0.9770538
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9778380, upper bound: 0.9770533
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9804905, upper bound: 0.9770541
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9791765, upper bound: 0.9770536
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9818288, upper bound: 0.9770539
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9805013, upper bound: 0.9770536
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9831537, upper bound: 0.9770544
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9703550, upper bound: 0.9831534
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9730160, upper bound: 0.9831535
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9717247, upper bound: 0.9831536
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9743857, upper bound: 0.9831536
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9730228, upper bound: 0.9831536
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.19
Output dim: 0, lower bound: -0.9756838, upper bound: 0.9831536
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 0, lower bound: -0.9778934, upper bound: 0.9840070
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 0, lower bound: -0.9738563, upper bound: 0.9840078
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 0, lower bound: -0.9752259, upper bound: 0.9840080
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 0, lower bound: -0.9765237, upper bound: 0.9840070
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 0, lower bound: -0.9778933, upper bound: 0.9840070
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.9054350852966309
rel_dist={0: [-0.9840186515659699, 0.9840187223222951]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 471

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8698154
time: 5.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8742890
time: 9.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.20
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8698154
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.20
Output dim: 0, lower bound: -0.8742861, upper bound: 0.8742890

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 6.6551013, 9.0135641, 6.6483841, 9.0208206, -1.7911229, 1.7899330
1: -17.4738808, -13.8005133, -17.4782696, -13.7856483, -2.6310883, 2.6196432
2: -3.2627931, -0.5239117, -3.2672892, -0.5199802, -2.3435087, 2.3436892
3: -10.8545074, -7.9736691, -10.8565407, -7.9592094, -2.4428225, 2.4299319
4: -12.5303125, -9.0268316, -12.5336456, -9.0226068, -2.7368693, 2.7304783
5: -4.9079676, -2.6729443, -4.9315715, -2.6704021, -1.8402195, 1.8614390
6: -3.0038805, -0.5669265, -3.0362353, -0.5638542, -2.0689967, 2.1008337
7: -9.3258791, -5.4602871, -9.3304529, -5.4337687, -2.9822502, 2.9613748
8: -2.5947652, -0.3486171, -2.5973587, -0.3460083, -2.0295730, 2.0295670
9: -4.4674053, -1.7695539, -4.4723735, -1.7615666, -2.2728467, 2.2686403

Time for backsubstitution: 12.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698188, upper bound: 0.8698164
time: 7.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8698159
time: 18.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.6375189, 9.0313349, -1.8851535, 1.8454707
1: -17.5395622, -13.7612896, -17.4851456, -13.7643452, -2.8484926, 2.7954130
2: -3.2866726, -0.4902350, -3.2758975, -0.5142211, -2.3659148, 2.3812411
3: -10.9079905, -7.9272232, -10.8677931, -7.9381194, -2.6484594, 2.6109834
4: -12.5829248, -8.9888010, -12.5387926, -9.0154037, -2.8057685, 2.7976289
5: -4.9724512, -2.5807762, -4.9653697, -2.6635838, -2.0558095, 2.0907950
6: -3.0983086, -0.4341698, -3.0826256, -0.5545936, -2.2864566, 2.3228679
7: -9.4304581, -5.3878284, -9.3434505, -5.3957224, -3.2355385, 3.1883574
8: -2.6340675, -0.3340945, -2.6018820, -0.3418722, -2.0750737, 2.0553896
9: -4.5135813, -1.7374017, -4.4801068, -1.7481221, -2.3314004, 2.3200221

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 471

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742867
time: 6.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742867
time: 6.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.47 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 26.47
Output dim: 0, lower bound: -0.8698188, upper bound: 0.8698164
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 26.47
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8698159
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742867
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 0, lower bound: -0.8698157, upper bound: 0.8742867

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 6.5945582, 9.0354347, 6.6551013, 9.0135641, -1.8634112, 1.8060627
1: -17.5384140, -13.7615099, -17.4738808, -13.8005133, -2.8112435, 2.6560755
2: -3.2863755, -0.4930611, -3.2627931, -0.5239117, -2.3559709, 2.3710477
3: -10.9072323, -7.9287009, -10.8545074, -7.9736691, -2.6075912, 2.4747338
4: -12.5824928, -8.9957676, -12.5303125, -9.0268316, -2.7818499, 2.7572041
5: -4.9721651, -2.5824437, -4.9079676, -2.6729443, -1.8909025, 2.0331190
6: -3.0973396, -0.4360595, -3.0038805, -0.5669265, -2.1323233, 2.2416036
7: -9.4290800, -5.3902311, -9.3258791, -5.4602871, -3.1710272, 3.0285301
8: -2.6290083, -0.3342519, -2.5947652, -0.3486171, -2.0629888, 2.0429232
9: -4.5133843, -1.7397047, -4.4674053, -1.7695539, -2.3064568, 2.2943473

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742833
time: 13.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698132, upper bound: 0.8742835
time: 5.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 6.5904884, 9.0354805, 6.5904884, 9.0354805, -1.8864167, 1.8864167
1: -17.5395622, -13.7612896, -17.5395622, -13.7612896, -2.8487988, 2.8487988
2: -3.2866726, -0.4902350, -3.2866726, -0.4902350, -2.3823915, 2.3823912
3: -10.9079905, -7.9272232, -10.9079905, -7.9272232, -2.6587739, 2.6587739
4: -12.5829248, -8.9888010, -12.5829248, -8.9888010, -2.8182054, 2.8182049
5: -4.9724512, -2.5807762, -4.9724512, -2.5807762, -2.0993719, 2.0993719
6: -3.0983086, -0.4341698, -3.0983086, -0.4341698, -2.3265626, 2.3265629
7: -9.4304581, -5.3878284, -9.4304581, -5.3878284, -3.2428918, 3.2428918
8: -2.6340675, -0.3340945, -2.6340675, -0.3340945, -2.0821414, 2.0821414
9: -4.5135813, -1.7374017, -4.5135813, -1.7374017, -2.3309526, 2.3309526

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742850
time: 13.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698132, upper bound: 0.8742869
time: 7.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 34.48 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 34.48
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742833
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 34.48
Output dim: 0, lower bound: -0.8698132, upper bound: 0.8742835
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.48
Output dim: 0, lower bound: -0.8678158, upper bound: 0.8742850
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.48
Output dim: 0, lower bound: -0.8698132, upper bound: 0.8742869

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 6.6008067, 9.0345583, 6.6581054, 9.0131464, -1.8564358, 1.8019683
1: -17.5293694, -13.7631664, -17.4695206, -13.8013039, -2.8012877, 2.6500416
2: -3.2809944, -0.4964572, -3.2601893, -0.5255146, -2.3490739, 2.3652980
3: -10.9038839, -7.9311013, -10.8528919, -7.9748173, -2.6028585, 2.4708724
4: -12.5798359, -9.0216742, -12.5290480, -9.0393143, -2.7665958, 2.7299187
5: -4.9655232, -2.5834088, -4.9047637, -2.6734011, -1.8827591, 2.0280495
6: -3.0877483, -0.4375153, -2.9992499, -0.5676165, -2.1219063, 2.2349854
7: -9.4262457, -5.4261031, -9.3245430, -5.4775710, -3.1479912, 2.9912100
8: -2.6183672, -0.3359671, -2.5896349, -0.3494496, -2.0512314, 2.0357397
9: -4.5112381, -1.7585905, -4.4663811, -1.7786601, -2.2937784, 2.2744827

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667432, upper bound: 0.8742821
time: 24.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678174, upper bound: 0.8742826
time: 10.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 6.5919037, 9.0498772, 6.6551085, 9.0135632, -1.8637614, 1.8165374
1: -17.5407639, -13.7462187, -17.4738693, -13.8005123, -2.8108015, 2.6715860
2: -3.2896268, -0.4825485, -3.2627907, -0.5239146, -2.3590307, 2.3767905
3: -10.9094744, -7.9161010, -10.8545074, -7.9736719, -2.6087379, 2.4901853
4: -12.6521320, -8.9946289, -12.5303087, -9.0268574, -2.8057406, 2.7455201
5: -4.9742489, -2.5712764, -4.9079614, -2.6729450, -1.8911214, 2.0352232
6: -3.0986755, -0.4161015, -3.0038753, -0.5669270, -2.1312218, 2.2462006
7: -9.5273170, -5.3893204, -9.3258781, -5.4603281, -3.1843920, 3.0143955
8: -2.6306009, -0.3083463, -2.5947595, -0.3486185, -2.0593395, 2.0490568
9: -4.5604935, -1.7375293, -4.4674048, -1.7695749, -2.3133614, 2.2886460

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687402, upper bound: 0.8742847
time: 10.71 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742821
time: 6.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 6.5967369, 9.0346031, 6.5935001, 9.0350609, -1.8794403, 1.8819028
1: -17.5305214, -13.7629414, -17.5352077, -13.7620773, -2.8388405, 2.8427653
2: -3.2812915, -0.4936304, -3.2840779, -0.4918520, -2.3754373, 2.3766146
3: -10.9046412, -7.9296236, -10.9063721, -7.9283738, -2.6540408, 2.6549635
4: -12.5802679, -9.0147047, -12.5816536, -9.0012798, -2.8029456, 2.7908797
5: -4.9658098, -2.5817404, -4.9692516, -2.5812387, -2.0913725, 2.0942960
6: -3.0887170, -0.4356251, -3.0936868, -0.4348655, -2.3163233, 2.3199451
7: -9.4276257, -5.4237018, -9.4291029, -5.4051099, -3.2198606, 3.2055614
8: -2.6234269, -0.3358107, -2.6289415, -0.3349171, -2.0703888, 2.0741284
9: -4.5114365, -1.7562879, -4.5125546, -1.7465055, -2.3198161, 2.3111010

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667401, upper bound: 0.8742819
time: 7.85 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8678144, upper bound: 0.8742836
time: 5.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 6.5878325, 9.0499220, 6.5904946, 9.0354795, -1.8867664, 1.8894377
1: -17.5419140, -13.7459955, -17.5395546, -13.7612896, -2.8483567, 2.8643079
2: -3.2899241, -0.4797205, -3.2866697, -0.4902371, -2.3853559, 2.3927445
3: -10.9102325, -7.9146242, -10.9079876, -7.9272251, -2.6599193, 2.6745429
4: -12.6525669, -8.9876604, -12.5829210, -8.9888268, -2.8451233, 2.8064966
5: -4.9745350, -2.5696204, -4.9724455, -2.5807774, -2.0994854, 2.1014755
6: -3.0996435, -0.4142132, -3.0983019, -0.4341712, -2.3255401, 2.3311596
7: -9.5286961, -5.3869095, -9.4304571, -5.3878689, -3.2562561, 3.2287941
8: -2.6356602, -0.3081894, -2.6340623, -0.3340960, -2.0784912, 2.0850003
9: -4.5606918, -1.7352251, -4.5135808, -1.7374227, -2.3413637, 2.3252058

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5778
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5778

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687402, upper bound: 0.8742833
time: 8.84 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742834
time: 5.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.58 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8667432, upper bound: 0.8742821
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8678174, upper bound: 0.8742826
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8687402, upper bound: 0.8742847
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742821
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8667401, upper bound: 0.8742819
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8678144, upper bound: 0.8742836
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8687402, upper bound: 0.8742833
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.58
Output dim: 0, lower bound: -0.8698118, upper bound: 0.8742834

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 6.6054144, 9.0301447, 6.6659698, 9.0047913, -1.8430796, 1.7906806
1: -17.5264778, -13.7699032, -17.4642944, -13.8141804, -2.7819748, 2.6385326
2: -3.2794778, -0.5012267, -3.2581165, -0.5339565, -2.3324819, 2.3513064
3: -10.8995094, -7.9325976, -10.8473883, -7.9772072, -2.5885854, 2.4596205
4: -12.5779047, -9.0254164, -12.5256100, -9.0449924, -2.7590833, 2.7227297
5: -4.9586725, -2.5856571, -4.8917737, -2.6765103, -1.8673258, 2.0121403
6: -3.0826046, -0.4406734, -2.9896944, -0.5719080, -2.1068511, 2.2209110
7: -9.4219875, -5.4347296, -9.3184404, -5.4939013, -3.1227174, 2.9740956
8: -2.6158257, -0.3411379, -2.5859175, -0.3592796, -2.0377960, 2.0268722
9: -4.5088511, -1.7640600, -4.4620347, -1.7879099, -2.2793486, 2.2624176

Time for backsubstitution: 12.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637476, upper bound: 0.8731452
time: 5.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8656722, upper bound: 0.8731454
time: 5.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 6.6001115, 9.0345631, 6.6358743, 9.0180902, -1.8609056, 1.8230858
1: -17.5298634, -13.7631416, -17.5136700, -13.7994499, -2.8013678, 2.6952152
2: -3.2811115, -0.4960325, -3.2811527, -0.5143400, -2.3618629, 2.3754210
3: -10.9041948, -7.9308743, -10.8593168, -7.9577312, -2.6150341, 2.4779425
4: -12.5799904, -9.0206547, -12.5573940, -9.0336246, -2.7710142, 2.7599692
5: -4.9655609, -2.5826607, -4.9155097, -2.6459894, -1.8871176, 2.0369816
6: -3.0879087, -0.4366608, -3.0143089, -0.5453429, -2.1247106, 2.2497380
7: -9.4268408, -5.4258761, -9.3557730, -5.4736080, -3.1506495, 3.0028780
8: -2.6191392, -0.3359251, -2.6104531, -0.3391390, -2.0588937, 2.0465934
9: -4.5113163, -1.7582481, -4.4893188, -1.7723823, -2.2997687, 2.2970088

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647877, upper bound: 0.8731485
time: 7.26 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667152, upper bound: 0.8731474
time: 6.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 6.5965056, 9.0454664, 6.6629725, 9.0052090, -1.8504105, 1.8049226
1: -17.5378742, -13.7529545, -17.4686432, -13.8133926, -2.7914848, 2.6600771
2: -3.2881031, -0.4873297, -3.2607160, -0.5323590, -2.3425112, 2.3622863
3: -10.9051018, -7.9175901, -10.8490009, -7.9760637, -2.5944672, 2.4789310
4: -12.6501932, -8.9983692, -12.5268726, -9.0325346, -2.7979529, 2.7383344
5: -4.9673967, -2.5735512, -4.8949704, -2.6760573, -1.8756866, 2.0193076
6: -3.0935345, -0.4192629, -2.9943199, -0.5712204, -2.1161633, 2.2321315
7: -9.5230350, -5.3979459, -9.3197746, -5.4766593, -3.1591206, 2.9972816
8: -2.6280560, -0.3135138, -2.5910406, -0.3584447, -2.0459013, 2.0393751
9: -4.5581017, -1.7430017, -4.4630542, -1.7788320, -2.2989507, 2.2765923

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8657438, upper bound: 0.8731487
time: 9.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8676709, upper bound: 0.8731459
time: 15.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 6.5912089, 9.0498829, 6.6328712, 9.0185108, -1.8682346, 1.8306091
1: -17.5412579, -13.7461958, -17.5180149, -13.7986603, -2.8108811, 2.7166727
2: -3.2897439, -0.4821233, -3.2837651, -0.5127349, -2.3717959, 2.3864102
3: -10.9097853, -7.9158731, -10.8609295, -7.9565716, -2.6209221, 2.4972551
4: -12.6522846, -8.9936104, -12.5586548, -9.0211687, -2.8100917, 2.7755656
5: -4.9742861, -2.5705295, -4.9187050, -2.6455319, -1.8954799, 2.0441694
6: -3.0988371, -0.4152479, -3.0189333, -0.5446553, -2.1340256, 2.2609568
7: -9.5279121, -5.3890824, -9.3571053, -5.4563665, -3.1870494, 3.0260663
8: -2.6313734, -0.3083053, -2.6155753, -0.3383121, -2.0669971, 2.0574129
9: -4.5605707, -1.7371857, -4.4903374, -1.7632993, -2.3193555, 2.3111939

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667885, upper bound: 0.8731458
time: 5.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687124, upper bound: 0.8731467
time: 8.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 6.6008444, 9.0302010, 6.6014585, 9.0266190, -1.8664565, 1.8692232
1: -17.5277710, -13.7696524, -17.5298786, -13.7749538, -2.8196683, 2.8275208
2: -3.2799578, -0.4980824, -3.2814846, -0.5002944, -2.3587193, 2.3630466
3: -10.9008236, -7.9309559, -10.8990746, -7.9309454, -2.6398959, 2.6395617
4: -12.5784121, -9.0177078, -12.5780649, -9.0070438, -2.7956591, 2.7843914
5: -4.9590054, -2.5836425, -4.9562006, -2.5849266, -2.0792532, 2.0789342
6: -3.0836873, -0.4383359, -3.0840569, -0.4401159, -2.3041315, 2.3063006
7: -9.4238663, -5.4322658, -9.4216881, -5.4215488, -3.1951194, 3.1848450
8: -2.6214495, -0.3409424, -2.6250887, -0.3447623, -2.0572085, 2.0643015
9: -4.5091109, -1.7613885, -4.5080237, -1.7562952, -2.3048716, 2.2995663

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637479, upper bound: 0.8731471
time: 11.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8656722, upper bound: 0.8731492
time: 7.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 6.5967436, 9.0345936, 6.5709510, 9.0399084, -1.8833370, 1.8957177
1: -17.5305176, -13.7629623, -17.5794239, -13.7602892, -2.8383002, 2.8792572
2: -3.2812893, -0.4936373, -3.3052549, -0.4805888, -2.3866758, 2.3920798
3: -10.9046354, -7.9296265, -10.9131994, -7.9112334, -2.6657863, 2.6681833
4: -12.5802650, -9.0147114, -12.6099625, -8.9966173, -2.8071580, 2.8200498
5: -4.9657969, -2.5817440, -4.9799042, -2.5539083, -2.1004705, 2.1024351
6: -3.0887055, -0.4356298, -3.1083202, -0.4130559, -2.3238435, 2.3336041
7: -9.4276190, -5.4237175, -9.4599428, -5.4011130, -3.2214880, 3.2177155
8: -2.6234241, -0.3358221, -2.6497669, -0.3246608, -2.0774779, 2.0822296
9: -4.5114307, -1.7562970, -4.5354939, -1.7406018, -2.3256683, 2.3308945

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647877, upper bound: 0.8731468
time: 12.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667123, upper bound: 0.8731479
time: 5.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 6.5919352, 9.0455198, 6.5984526, 9.0270376, -1.8737881, 1.8767618
1: -17.5391655, -13.7527027, -17.5342236, -13.7741652, -2.8291817, 2.8490644
2: -3.2885833, -0.4841846, -3.2840734, -0.4986801, -2.3687124, 2.3783891
3: -10.9064150, -7.9159498, -10.9006891, -7.9297967, -2.6457806, 2.6591372
4: -12.6507034, -8.9906616, -12.5793295, -8.9945889, -2.8374934, 2.8000069
5: -4.9677300, -2.5715361, -4.9593949, -2.5844674, -2.0873618, 2.0861089
6: -3.0946164, -0.4169250, -3.0886750, -0.4394221, -2.3133492, 2.3175206
7: -9.5249138, -5.3954697, -9.4230385, -5.4043064, -3.2315187, 3.2080762
8: -2.6336784, -0.3133197, -2.6302094, -0.3439407, -2.0653090, 2.0751762
9: -4.5583577, -1.7403307, -4.5090480, -1.7472166, -2.3263884, 2.3136806

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8657438, upper bound: 0.8731469
time: 8.12 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8676738, upper bound: 0.8731469
time: 8.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 6.5878410, 9.0499134, 6.5679374, 9.0403280, -1.8906674, 1.9032495
1: -17.5419064, -13.7460146, -17.5837708, -13.7594995, -2.8478146, 2.8877401
2: -3.2899203, -0.4797289, -3.3078632, -0.4789701, -2.3966122, 2.4031420
3: -10.9102259, -7.9146261, -10.9148130, -7.9100828, -2.6716747, 2.6877627
4: -12.6525612, -8.9876661, -12.6112318, -8.9841652, -2.8488269, 2.8356671
5: -4.9745235, -2.5696254, -4.9830976, -2.5534449, -2.1085739, 2.1096213
6: -3.0996330, -0.4142179, -3.1129363, -0.4123602, -2.3330603, 2.3448217
7: -9.5286884, -5.3869247, -9.4612932, -5.3838730, -3.2578831, 3.2409480
8: -2.6356564, -0.3082013, -2.6548867, -0.3238473, -2.0855770, 2.0931013
9: -4.5606871, -1.7352346, -4.5365219, -1.7315209, -2.3470092, 2.3450630

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 5773
type: A, layer: 1, pos: 5778
type: A, layer: 1, pos: 4603
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 859

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667884, upper bound: 0.8731462
time: 5.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8687124, upper bound: 0.8731460
time: 5.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.76 seconds
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8637476, upper bound: 0.8731452
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8656722, upper bound: 0.8731454
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8647877, upper bound: 0.8731485
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8667152, upper bound: 0.8731474
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8657438, upper bound: 0.8731487
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8676709, upper bound: 0.8731459
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8667885, upper bound: 0.8731458
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8687124, upper bound: 0.8731467
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8637479, upper bound: 0.8731471
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8656722, upper bound: 0.8731492
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8647877, upper bound: 0.8731468
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8667123, upper bound: 0.8731479
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8657438, upper bound: 0.8731469
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8676738, upper bound: 0.8731469
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8667884, upper bound: 0.8731462
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.76
Output dim: 0, lower bound: -0.8687124, upper bound: 0.8731460

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 6.6161118, 9.0243511, 6.6711106, 9.0036879, -1.8309469, 1.7792943
1: -17.5187340, -13.7751312, -17.4609756, -13.8163691, -2.7712078, 2.6224484
2: -3.2737226, -0.5121232, -3.2563319, -0.5392926, -2.3202233, 2.3380246
3: -10.8950872, -7.9412065, -10.8453550, -7.9812241, -2.5768499, 2.4454012
4: -12.5638609, -9.0306053, -12.5189743, -9.0462532, -2.7436337, 2.7108474
5: -4.9546614, -2.5892200, -4.8899856, -2.6774549, -1.8599343, 2.0053778
6: -3.0705621, -0.4467010, -2.9833534, -0.5730739, -2.0925474, 2.2054291
7: -9.4112492, -5.4444795, -9.3157120, -5.4991236, -3.1043940, 2.9610050
8: -2.6018963, -0.3456316, -2.5784426, -0.3599467, -2.0240469, 2.0154285
9: -4.4980936, -1.7688158, -4.4565716, -1.7885803, -2.2666531, 2.2512338

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8637505, upper bound: 0.8711438
time: 6.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637476, upper bound: 0.8731452
time: 5.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 6.6054211, 9.0301447, 6.6659727, 9.0047922, -1.8389015, 1.7906744
1: -17.5264740, -13.7700958, -17.4642906, -13.8142748, -2.7708530, 2.6351104
2: -3.2794747, -0.5012374, -3.2581153, -0.5339626, -2.3324742, 2.3460195
3: -10.8995066, -7.9326048, -10.8473873, -7.9772100, -2.5891533, 2.4596066
4: -12.5778933, -9.0254183, -12.5256042, -9.0449944, -2.7588334, 2.7227225
5: -4.9586682, -2.5859294, -4.8917723, -2.6766438, -1.8682764, 2.0114570
6: -3.0825968, -0.4406757, -2.9896891, -0.5719094, -2.1015983, 2.2183981
7: -9.4219818, -5.4347372, -9.3184385, -5.4939060, -3.1199498, 2.9714632
8: -2.6158166, -0.3411388, -2.5859127, -0.3592792, -2.0310450, 2.0259433
9: -4.5088434, -1.7640612, -4.4620304, -1.7879088, -2.2753563, 2.2624133

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8656722, upper bound: 0.8711438
time: 5.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8656722, upper bound: 0.8731454
time: 5.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 6.6108007, 9.0287695, 6.6410370, 9.0169878, -1.8487659, 1.8101075
1: -17.5221214, -13.7683706, -17.5103436, -13.8016357, -2.7906046, 2.6791158
2: -3.2753520, -0.5069377, -3.2793665, -0.5196773, -2.3481460, 2.3621433
3: -10.8997707, -7.9394836, -10.8572817, -7.9617357, -2.6033335, 2.4637141
4: -12.5659409, -9.0258427, -12.5507565, -9.0348835, -2.7555628, 2.7481048
5: -4.9615498, -2.5862281, -4.9137030, -2.6469374, -1.8797235, 2.0301967
6: -3.0758703, -0.4426899, -3.0079787, -0.5465145, -2.1104007, 2.2342565
7: -9.4161043, -5.4356408, -9.3530445, -5.4788332, -3.1323137, 2.9897850
8: -2.6052122, -0.3404164, -2.6029730, -0.3398104, -2.0451331, 2.0336194
9: -4.5005550, -1.7630019, -4.4838195, -1.7730674, -2.2870865, 2.2847519

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8647877, upper bound: 0.8711468
time: 6.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647877, upper bound: 0.8731485
time: 7.30 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 6.6001177, 9.0345612, 6.6358786, 9.0180893, -1.8567328, 1.8206694
1: -17.5298615, -13.7633324, -17.5136681, -13.7995453, -2.7902460, 2.6917930
2: -3.2811098, -0.4960427, -3.2811520, -0.5143442, -2.3597598, 2.3701448
3: -10.9041920, -7.9308815, -10.8593168, -7.9577322, -2.6156015, 2.4779296
4: -12.5799789, -9.0206566, -12.5573893, -9.0336256, -2.7707682, 2.7599635
5: -4.9655581, -2.5829310, -4.9155068, -2.6461265, -1.8880851, 2.0362890
6: -3.0879006, -0.4366627, -3.0143058, -0.5453439, -2.1194596, 2.2472248
7: -9.4268360, -5.4258819, -9.3557711, -5.4736128, -3.1478853, 3.0002503
8: -2.6191330, -0.3359256, -2.6104484, -0.3391395, -2.0521402, 2.0439835
9: -4.5113077, -1.7582492, -4.4893131, -1.7723827, -2.2957754, 2.2955523

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8667124, upper bound: 0.8711456
time: 5.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8667124, upper bound: 0.8731474
time: 5.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 6.6071954, 9.0396748, 6.6681128, 9.0041065, -1.8382781, 1.7919717
1: -17.5301285, -13.7581816, -17.4653244, -13.8155804, -2.7807202, 2.6439934
2: -3.2823634, -0.4982394, -3.2589335, -0.5376946, -2.3302011, 2.3489907
3: -10.9006767, -7.9262199, -10.8469667, -7.9800839, -2.5827303, 2.4646873
4: -12.6361256, -9.0035572, -12.5202332, -9.0337963, -2.7825346, 2.7264543
5: -4.9633889, -2.5771146, -4.8931832, -2.6770015, -1.8682981, 2.0125396
6: -3.0814908, -0.4252930, -2.9879799, -0.5723877, -2.1018572, 2.2166655
7: -9.5122356, -5.4076900, -9.3170395, -5.4818802, -3.1408153, 2.9841888
8: -2.6141281, -0.3180084, -2.5835657, -0.3591123, -2.0321531, 2.0264146
9: -4.5473261, -1.7477573, -4.4575911, -1.7794997, -2.2862470, 2.2654061

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8657437, upper bound: 0.8711447
time: 9.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8657440, upper bound: 0.8711447
time: 9.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 6.5965095, 9.0454645, 6.6629763, 9.0052080, -1.8462327, 1.8025100
1: -17.5378723, -13.7531452, -17.4686413, -13.8134871, -2.7803659, 2.6566558
2: -3.2881019, -0.4873395, -3.2607150, -0.5323648, -2.3425031, 2.3569996
3: -10.9050989, -7.9175968, -10.8490009, -7.9760666, -2.5950351, 2.4789171
4: -12.6501837, -8.9983721, -12.5268669, -9.0325356, -2.7966146, 2.7383287
5: -4.9673948, -2.5738249, -4.8949695, -2.6761913, -1.8766370, 2.0186298
6: -3.0935268, -0.4192653, -2.9943151, -0.5712218, -2.1109111, 2.2296267
7: -9.5230331, -5.3979521, -9.3197727, -5.4766626, -3.1563730, 2.9946482
8: -2.6280489, -0.3135147, -2.5910358, -0.3584452, -2.0391512, 2.0367684
9: -4.5580940, -1.7430027, -4.4630504, -1.7788312, -2.2949591, 2.2765875

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8676707, upper bound: 0.8711446
time: 7.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8676710, upper bound: 0.8711426
time: 15.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 6.6018939, 9.0440922, 6.6380320, 9.0174074, -1.8560953, 1.8176291
1: -17.5335121, -13.7514210, -17.5146904, -13.8008461, -2.8001165, 2.7006621
2: -3.2840014, -0.4930393, -3.2819786, -0.5180731, -2.3580921, 2.3731191
3: -10.9053602, -7.9245081, -10.8588963, -7.9605794, -2.6092401, 2.4830055
4: -12.6382122, -8.9987965, -12.5520172, -9.0224285, -2.7946687, 2.7637043
5: -4.9702764, -2.5740986, -4.9169002, -2.6464789, -1.8880885, 2.0373783
6: -3.0867972, -0.4212813, -3.0126009, -0.5458269, -2.1197135, 2.2454934
7: -9.5171146, -5.3988452, -9.3543768, -5.4615922, -3.1687412, 3.0129709
8: -2.6174440, -0.3127961, -2.6080947, -0.3389864, -2.0532379, 2.0444441
9: -4.5497947, -1.7419406, -4.4848442, -1.7639813, -2.3066621, 2.2989373

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8667883, upper bound: 0.8711418
time: 5.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8667881, upper bound: 0.8711416
time: 5.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 6.5912151, 9.0498819, 6.6328735, 9.0185099, -1.8640618, 1.8281960
1: -17.5412540, -13.7463837, -17.5180149, -13.7987576, -2.7997603, 2.7132354
2: -3.2897418, -0.4821341, -3.2837634, -0.5127392, -2.3696995, 2.3811343
3: -10.9097834, -7.9158802, -10.8609276, -7.9565744, -2.6214905, 2.4972425
4: -12.6522760, -8.9936113, -12.5586491, -9.0211678, -2.8087537, 2.7755601
5: -4.9742842, -2.5708025, -4.9187040, -2.6456692, -1.8964474, 2.0434799
6: -3.0988276, -0.4152498, -3.0189288, -0.5446548, -2.1287739, 2.2584510
7: -9.5279083, -5.3890896, -9.3571053, -5.4563704, -3.1843042, 3.0234361
8: -2.6313658, -0.3083053, -2.6155715, -0.3383131, -2.0602446, 2.0548074
9: -4.5605640, -1.7371864, -4.4903340, -1.7633009, -2.3153610, 2.3097374

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5773
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 4603
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5773

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8442518711090088
rel_dist={0: [-0.8742898229747036, 0.874290832632278]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2431.68 seconds
