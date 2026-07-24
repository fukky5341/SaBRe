## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.67154946715
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8134632, -6.7279902, -8.8134632, -6.7279902, -2.0854731, 2.0854731)
1: (1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.6906104, 1.6906104)
2: (-5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.5822163, 1.5822163)
3: (-10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.5740218, 1.5740218)
4: (-4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704)
5: (-8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.5837355, 1.5837355)
6: (-5.9832397, -3.9410968, -5.9832397, -3.9410968, -2.0421429, 2.0421429)
7: (-4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261)
8: (-3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.4418497, 1.4418497)
9: (-11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.9131136, 1.9131136)

## BASE Result
execution time: IAR + LP analysis = 15.23 + 31.83 = 47.06 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.94 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.4624950885772705
rel_dist={1: [-0.9108364634088741, 0.9108356417165169]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.2293808460235596
rel_dist={1: [-0.49714962017070174, 0.49714935310829933]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.2760038375854492
rel_dist={1: [-0.5884148639986186, 0.5884165441500335]}

## Binary Search Result
Binary search time: 197.87 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3355.07 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849460
time: 3.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849392, upper bound: 0.9849404
time: 3.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849460
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 1, lower bound: -0.9849392, upper bound: 0.9849404

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8134632, -6.7279902, -1.8962209, 1.8830910
1: 1.9655743, 3.6435642, 1.9534850, 3.6440954, -1.4964547, 1.5083833
2: -5.4579616, -3.8526118, -5.4696188, -3.8512685, -1.4084828, 1.4187267
3: -10.1621323, -8.4082584, -10.1650553, -8.4066534, -1.2322814, 1.2348394
4: -4.7673235, -3.3324428, -4.7845268, -3.3316565, -1.4356670, 1.4520841
5: -8.3700380, -6.7941961, -8.3735094, -6.7897739, -1.3867939, 1.3874466
6: -5.9793115, -3.9414420, -5.9832397, -3.9410968, -1.8306606, 1.8331709
7: -4.2065864, -2.8140006, -4.2095613, -2.8125353, -1.3940511, 1.3955607
8: -3.7371349, -2.3019557, -3.7387199, -2.2968702, -1.3114526, 1.3075582
9: -11.0497036, -9.1602478, -11.0502615, -9.1371479, -1.6582141, 1.6359031

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9758259
time: 3.91 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849404
time: 3.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8134604, -6.7280040, -1.9163032, 1.9241607
1: 1.9334784, 3.7011304, 1.9534936, 3.6440954, -1.5318940, 1.5248306
2: -5.4810061, -3.7874212, -5.4696097, -3.8512692, -1.4385879, 1.4385879
3: -10.1747808, -8.3837261, -10.1650505, -8.4066563, -1.2543864, 1.2519886
4: -4.7960691, -3.2351460, -4.7845116, -3.3316579, -1.4644113, 1.5493655
5: -8.3994455, -6.7868471, -8.3735075, -6.7897768, -1.4088588, 1.4036362
6: -5.9932337, -3.9284849, -5.9832363, -3.9410973, -1.8471017, 1.8456520
7: -4.2181315, -2.7841961, -4.2095566, -2.8125367, -1.4055948, 1.4253604
8: -3.7703476, -2.2896943, -3.7387180, -2.2968731, -1.3353868, 1.3365424
9: -11.1756716, -9.1263027, -11.0502605, -9.1371651, -1.6821744, 1.6871430

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849417, upper bound: 0.9758262
time: 3.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849417, upper bound: 0.9849408
time: 3.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9758259
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849404
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 1, lower bound: -0.9849417, upper bound: 0.9758262
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 1, lower bound: -0.9849417, upper bound: 0.9849408

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8123894, -6.7421303, -1.8818927, 1.8818924
1: 1.9655743, 3.6435642, 1.9655743, 3.6435642, -1.4957201, 1.4957200
2: -5.4579616, -3.8526118, -5.4579616, -3.8526118, -1.4072270, 1.4072268
3: -10.1621323, -8.4082584, -10.1621323, -8.4082584, -1.2303560, 1.2303560
4: -4.7673235, -3.3324428, -4.7673235, -3.3324428, -1.4348807, 1.4348807
5: -8.3700380, -6.7941961, -8.3700380, -6.7941961, -1.3818145, 1.3818145
6: -5.9793115, -3.9414420, -5.9793115, -3.9414420, -1.8300614, 1.8300612
7: -4.2065864, -2.8140006, -4.2065864, -2.8140006, -1.3925858, 1.3925858
8: -3.7371349, -2.3019557, -3.7371349, -2.3019557, -1.3054949, 1.3054950
9: -11.0497036, -9.1602478, -11.0497036, -9.1602478, -1.6351233, 1.6351228

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9640695, upper bound: 0.9416860
time: 3.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758240, upper bound: 0.9758355
time: 3.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8982000, -6.7152944, -1.9081860, 1.9019432
1: 1.9655743, 3.6435642, 1.9334784, 3.7011304, -1.5121124, 1.5259328
2: -5.4579616, -3.8526118, -5.4810061, -3.7874212, -1.4270375, 1.4303017
3: -10.1621323, -8.4082584, -10.1747808, -8.3837261, -1.2475247, 1.2421081
4: -4.7673235, -3.3324428, -4.7960691, -3.2351460, -1.5321774, 1.4636264
5: -8.3700380, -6.7941961, -8.3994455, -6.7868471, -1.3893595, 1.4038752
6: -5.9793115, -3.9414420, -5.9932337, -3.9284849, -1.8425455, 1.8453956
7: -4.2065864, -2.8140006, -4.2181315, -2.7841961, -1.4223902, 1.4041309
8: -3.7371349, -2.3019557, -3.7703476, -2.2896943, -1.3214726, 1.3294003
9: -11.0497036, -9.1602478, -11.1756716, -9.1263027, -1.6680374, 1.6590936

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9640695, upper bound: 0.9508070
time: 3.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758240, upper bound: 0.9849429
time: 3.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8123894, -6.7421303, -1.9019432, 1.9081862
1: 1.9334784, 3.7011304, 1.9655743, 3.6435642, -1.5259328, 1.5121124
2: -5.4810061, -3.7874212, -5.4579616, -3.8526118, -1.4303017, 1.4270380
3: -10.1747808, -8.3837261, -10.1621323, -8.4082584, -1.2421076, 1.2475246
4: -4.7960691, -3.2351460, -4.7673235, -3.3324428, -1.4636264, 1.5321774
5: -8.3994455, -6.7868471, -8.3700380, -6.7941961, -1.4038751, 1.3893595
6: -5.9932337, -3.9284849, -5.9793115, -3.9414420, -1.8453956, 1.8425455
7: -4.2181315, -2.7841961, -4.2065864, -2.8140006, -1.4041309, 1.4223902
8: -3.7703476, -2.2896943, -3.7371349, -2.3019557, -1.3294003, 1.3214723
9: -11.1756716, -9.1263027, -11.0497036, -9.1602478, -1.6590934, 1.6680372

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9731852, upper bound: 0.9416766
time: 3.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849370, upper bound: 0.9758230
time: 3.90 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8982000, -6.7152944, -1.9287877, 1.9287877
1: 1.9334784, 3.7011304, 1.9334784, 3.7011304, -1.5424306, 1.5424304
2: -5.4810061, -3.7874212, -5.4810061, -3.7874212, -1.4499924, 1.4499924
3: -10.1747808, -8.3837261, -10.1747808, -8.3837261, -1.2606111, 1.2606112
4: -4.7960691, -3.2351460, -4.7960691, -3.2351460, -1.5609231, 1.5609231
5: -8.3994455, -6.7868471, -8.3994455, -6.7868471, -1.4113574, 1.4113575
6: -5.9932337, -3.9284849, -5.9932337, -3.9284849, -1.8563273, 1.8563272
7: -4.2181315, -2.7841961, -4.2181315, -2.7841961, -1.4339354, 1.4339354
8: -3.7703476, -2.2896943, -3.7703476, -2.2896943, -1.3433198, 1.3433195
9: -11.1756716, -9.1263027, -11.1756716, -9.1263027, -1.6923931, 1.6923932

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9731863, upper bound: 0.9416764
time: 3.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849381, upper bound: 0.9758643
time: 3.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.67 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9640695, upper bound: 0.9416860
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9758240, upper bound: 0.9758355
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9640695, upper bound: 0.9508070
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9758240, upper bound: 0.9849429
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9731852, upper bound: 0.9416766
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9849370, upper bound: 0.9758230
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9731863, upper bound: 0.9416764
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9849381, upper bound: 0.9758643

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8080835, -6.7458239, -1.8549814, 1.8402443
1: 2.0097160, 3.5735288, 1.9720440, 3.6320763, -1.3785243, 1.4182698
2: -5.4477758, -3.8951197, -5.4562645, -3.8598168, -1.3529115, 1.3591161
3: -10.1518946, -8.4342260, -10.1604261, -8.4160604, -1.2099364, 1.2025733
4: -4.7095294, -3.3394604, -4.7553225, -3.3335886, -1.3759408, 1.4158621
5: -8.2896528, -6.8105326, -8.3562202, -6.7966118, -1.2932217, 1.3239630
6: -5.9173589, -4.0664520, -5.9702334, -3.9617968, -1.6677580, 1.6978943
7: -4.1273794, -2.8371000, -4.1936898, -2.8174853, -1.3098941, 1.2842054
8: -3.7275267, -2.3217211, -3.7351398, -2.3053474, -1.2910098, 1.2802325
9: -10.9146385, -9.1954670, -11.0276880, -9.1652317, -1.4995146, 1.4215599

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9416905
time: 3.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9416915
time: 3.84 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8123894, -6.7421303, -1.8818889, 1.9409270
1: 1.9655747, 3.6435614, 1.9655743, 3.6435642, -1.5204837, 1.4957172
2: -5.4579601, -3.8526134, -5.4579616, -3.8526118, -1.4118381, 1.4072247
3: -10.1621304, -8.4082613, -10.1621323, -8.4082584, -1.2295303, 1.2701496
4: -4.7673187, -3.3324428, -4.7673235, -3.3324428, -1.4348760, 1.4348807
5: -8.3700352, -6.7941961, -8.3700380, -6.7941961, -1.3765123, 1.3818138
6: -5.9793115, -3.9414463, -5.9793115, -3.9414420, -1.8556788, 1.8300579
7: -4.2065811, -2.8140025, -4.2065864, -2.8140006, -1.3925805, 1.3925838
8: -3.7371349, -2.3019562, -3.7371349, -2.3019557, -1.3078308, 1.3054943
9: -11.0496979, -9.1602507, -11.0497036, -9.1602478, -1.6351175, 1.6422068

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9640873
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9758403
time: 3.91 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8939877, -6.7208433, -1.8812594, 1.8539553
1: 2.0097160, 3.5735288, 1.9399471, 3.6896782, -1.3805642, 1.4486911
2: -5.4477758, -3.8951197, -5.4793591, -3.7945230, -1.3633473, 1.3821695
3: -10.1518946, -8.4342260, -10.1730976, -8.3914833, -1.2256298, 1.2139719
4: -4.7095294, -3.3394604, -4.7839217, -3.2362547, -1.4732747, 1.4444613
5: -8.2896528, -6.8105326, -8.3859758, -6.7891879, -1.3006797, 1.3361328
6: -5.9173589, -4.0664520, -5.9841223, -3.9488535, -1.6741664, 1.7132010
7: -4.1273794, -2.8371000, -4.2052584, -2.7883868, -1.3389926, 1.2952371
8: -3.7275267, -2.3217211, -3.7683139, -2.2930851, -1.3061167, 1.3046262
9: -10.9146385, -9.1954670, -11.1536846, -9.1308994, -1.5323286, 1.4257565

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416777, upper bound: 0.9508061
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416765, upper bound: 0.9508053
time: 4.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8982000, -6.7152944, -1.9081831, 1.9566009
1: 1.9655747, 3.6435614, 1.9334784, 3.7011304, -1.5298083, 1.5259299
2: -5.4579601, -3.8526134, -5.4810061, -3.7874212, -1.4293139, 1.4302995
3: -10.1621304, -8.4082613, -10.1747808, -8.3837261, -1.2467270, 1.2819952
4: -4.7673187, -3.3324428, -4.7960691, -3.2351460, -1.5321727, 1.4636264
5: -8.3700352, -6.7941961, -8.3994455, -6.7868471, -1.3840578, 1.4005395
6: -5.9793115, -3.9414463, -5.9932337, -3.9284849, -1.8677220, 1.8453925
7: -4.2065811, -2.8140025, -4.2181315, -2.7841961, -1.4223850, 1.4041290
8: -3.7371349, -2.3019562, -3.7703476, -2.2896943, -1.3238077, 1.3293996
9: -11.0496979, -9.1602507, -11.1756716, -9.1263027, -1.6680326, 1.6548495

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416765, upper bound: 0.9731919
time: 3.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416777, upper bound: 0.9849441
time: 3.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8080835, -6.7458239, -1.8741622, 1.8640270
1: 1.9800858, 3.6309133, 1.9720440, 3.6320763, -1.4069247, 1.4345114
2: -5.4694366, -3.8297918, -5.4562645, -3.8598168, -1.3754697, 1.3768182
3: -10.1643639, -8.4138031, -10.1604261, -8.4160604, -1.2211354, 1.2131913
4: -4.7350035, -3.2420909, -4.7553225, -3.3335886, -1.4014149, 1.5132315
5: -8.3198881, -6.8043079, -8.3562202, -6.7966118, -1.3145261, 1.3306705
6: -5.9275594, -4.0536022, -5.9702334, -3.9617968, -1.6779237, 1.7101297
7: -4.1388626, -2.8128054, -4.1936898, -2.8174853, -1.3213773, 1.2919699
8: -3.7589579, -2.3098330, -3.7351398, -2.3053474, -1.3103499, 1.2958889
9: -11.0403852, -9.1633463, -11.0276880, -9.1652317, -1.5225282, 1.4572198

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508058, upper bound: 0.9416764
time: 3.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508058, upper bound: 0.9416760
time: 4.02 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8123894, -6.7421303, -1.9019432, 1.9672117
1: 1.9334812, 3.7011271, 1.9655743, 3.6435642, -1.5506959, 1.5121121
2: -5.4810052, -3.7874231, -5.4579616, -3.8526118, -1.4349127, 1.4270370
3: -10.1747808, -8.3837280, -10.1621323, -8.4082584, -1.2413838, 1.2832794
4: -4.7960649, -3.2351470, -4.7673235, -3.3324428, -1.4636221, 1.5321765
5: -8.3994436, -6.7868481, -8.3700380, -6.7941961, -1.3985729, 1.3893585
6: -5.9932308, -3.9284909, -5.9793115, -3.9414420, -1.8710120, 1.8425422
7: -4.2181268, -2.7841980, -4.2065864, -2.8140006, -1.4041262, 1.4223883
8: -3.7703452, -2.2896948, -3.7371349, -2.3019557, -1.3308690, 1.3214707
9: -11.1756668, -9.1263037, -11.0497036, -9.1602478, -1.6590934, 1.6751213

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508057, upper bound: 0.9640682
time: 3.64 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508056, upper bound: 0.9758254
time: 3.66 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8939877, -6.7208433, -1.9009204, 1.8788345
1: 1.9800858, 3.6309133, 1.9399471, 3.6896782, -1.4089646, 1.4648554
2: -5.4694366, -3.8297918, -5.4793591, -3.7945230, -1.3859155, 1.3997698
3: -10.1643639, -8.4138031, -10.1730976, -8.3914833, -1.2381120, 1.2247155
4: -4.7350035, -3.2420909, -4.7839217, -3.2362547, -1.4987488, 1.5418308
5: -8.3198881, -6.8043079, -8.3859758, -6.7891879, -1.3208892, 1.3454795
6: -5.9275594, -4.0536022, -5.9841223, -3.9488535, -1.6872799, 1.7233644
7: -4.1388626, -2.8128054, -4.2052584, -2.7883868, -1.3504758, 1.3038881
8: -3.7589579, -2.3098330, -3.7683139, -2.2930851, -1.3256949, 1.3180715
9: -11.0403852, -9.1633463, -11.1536846, -9.1308994, -1.5557647, 1.4645098

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9416764
time: 3.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9416761
time: 3.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8982000, -6.7152944, -1.9287872, 1.9836769
1: 1.9334812, 3.7011271, 1.9334784, 3.7011304, -1.5604732, 1.5424299
2: -5.4810052, -3.7874231, -5.4810061, -3.7874212, -1.4524300, 1.4499915
3: -10.1747808, -8.3837280, -10.1747808, -8.3837261, -1.2599165, 1.2964687
4: -4.7960649, -3.2351470, -4.7960691, -3.2351460, -1.5609188, 1.5609221
5: -8.3994436, -6.7868481, -8.3994455, -6.7868471, -1.4060578, 1.4102116
6: -5.9932308, -3.9284909, -5.9932337, -3.9284849, -1.8819389, 1.8563238
7: -4.2181268, -2.7841980, -4.2181315, -2.7841961, -1.4339306, 1.4339335
8: -3.7703452, -2.2896948, -3.7703476, -2.2896943, -1.3456564, 1.3433187
9: -11.1756668, -9.1263037, -11.1756716, -9.1263027, -1.6923926, 1.6887317

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9640683
time: 3.61 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508012, upper bound: 0.9758659
time: 3.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9416905
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9416915
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9640873
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416915, upper bound: 0.9758403
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416777, upper bound: 0.9508061
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416765, upper bound: 0.9508053
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416765, upper bound: 0.9731919
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9416777, upper bound: 0.9849441
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508058, upper bound: 0.9416764
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508058, upper bound: 0.9416760
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508057, upper bound: 0.9640682
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508056, upper bound: 0.9758254
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9416764
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9416761
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9640683
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.24
Output dim: 1, lower bound: -0.9508012, upper bound: 0.9758659

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.7869673, -6.7579565, -1.8212051, 1.8212049
1: 2.0097160, 3.5735288, 2.0097160, 3.5735288, -1.3433619, 1.3433619
2: -5.4477758, -3.8951197, -5.4477758, -3.8951197, -1.3226519, 1.3226519
3: -10.1518946, -8.4342260, -10.1518946, -8.4342260, -1.1935779, 1.1935776
4: -4.7095294, -3.3394604, -4.7095294, -3.3394604, -1.3700690, 1.3700690
5: -8.2896528, -6.8105326, -8.2896528, -6.8105326, -1.2677438, 1.2677439
6: -5.9173589, -4.0664520, -5.9173589, -4.0664520, -1.6113915, 1.6113917
7: -4.1273794, -2.8371000, -4.1273794, -2.8371000, -1.2332876, 1.2332873
8: -3.7275267, -2.3217211, -3.7275267, -2.3217211, -1.2725646, 1.2725645
9: -10.9146385, -9.1954670, -10.9146385, -9.1954670, -1.3550441, 1.3550440

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9363093, upper bound: 0.9407854
time: 3.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9416846
time: 3.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8123770, -6.7423153, -1.8533511, 1.8460727
1: 2.0097160, 3.5735288, 1.9656744, 3.6435490, -1.3790305, 1.4233352
2: -5.4477758, -3.8951197, -5.4579010, -3.8526344, -1.3616314, 1.3597338
3: -10.1518946, -8.4342260, -10.1621189, -8.4084911, -1.2174536, 1.2045596
4: -4.7095294, -3.3394604, -4.7671833, -3.3324485, -1.3770809, 1.4277229
5: -8.2896528, -6.8105326, -8.3699846, -6.7942533, -1.2950170, 1.3303525
6: -5.9173589, -4.0664520, -5.9791684, -3.9414470, -1.6682272, 1.7038622
7: -4.1273794, -2.8371000, -4.2065754, -2.8141119, -1.3132675, 1.2853746
8: -3.7275267, -2.3217211, -3.7370596, -2.3019710, -1.2947791, 1.2822757
9: -10.9146385, -9.1954670, -11.0496855, -9.1604042, -1.5009019, 1.4230793

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9363093, upper bound: 0.9407850
time: 3.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9416857
time: 3.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8123770, -6.7423153, -8.7869673, -6.7579565, -1.8460727, 1.8533509
1: 1.9656744, 3.6435490, 2.0097160, 3.5735288, -1.4233348, 1.3790305
2: -5.4579010, -3.8526344, -5.4477758, -3.8951197, -1.3597338, 1.3616312
3: -10.1621189, -8.4084911, -10.1518946, -8.4342260, -1.2045599, 1.2174536
4: -4.7671833, -3.3324485, -4.7095294, -3.3394604, -1.4277229, 1.3770809
5: -8.3699846, -6.7942533, -8.2896528, -6.8105326, -1.3303525, 1.2950171
6: -5.9791684, -3.9414470, -5.9173589, -4.0664520, -1.7038622, 1.6682277
7: -4.2065754, -2.8141119, -4.1273794, -2.8371000, -1.2853744, 1.3132675
8: -3.7370596, -2.3019710, -3.7275267, -2.3217211, -1.2822757, 1.2947789
9: -11.0496855, -9.1604042, -10.9146385, -9.1954670, -1.4230793, 1.5009017

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9363093, upper bound: 0.9631811
time: 3.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9640801
time: 3.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8123884, -6.7421303, -1.9409246, 1.9409249
1: 1.9655747, 3.6435614, 1.9655747, 3.6435614, -1.5204806, 1.5204809
2: -5.4579601, -3.8526134, -5.4579601, -3.8526134, -1.4118354, 1.4118357
3: -10.1621304, -8.4082613, -10.1621304, -8.4082613, -1.2701478, 1.2701479
4: -4.7673187, -3.3324428, -4.7673187, -3.3324428, -1.4348760, 1.4348760
5: -8.3700352, -6.7941961, -8.3700352, -6.7941961, -1.3765121, 1.3765118
6: -5.9793115, -3.9414463, -5.9793115, -3.9414463, -1.8556762, 1.8556757
7: -4.2065811, -2.8140025, -4.2065811, -2.8140025, -1.3925786, 1.3925786
8: -3.7371349, -2.3019562, -3.7371349, -2.3019562, -1.3078301, 1.3078301
9: -11.0496979, -9.1602507, -11.0496979, -9.1602507, -1.6422017, 1.6422017

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9363107, upper bound: 0.9749162
time: 3.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9758346
time: 3.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8731165, -6.7341380, -1.8449879, 1.8404906
1: 2.0097160, 3.5735288, 1.9800858, 3.6309133, -1.3563395, 1.3718992
2: -5.4477758, -3.8951197, -5.4694366, -3.8297918, -1.3407412, 1.3452098
3: -10.1518946, -8.4342260, -10.1643639, -8.4138031, -1.2055795, 1.2044647
4: -4.7095294, -3.3394604, -4.7350035, -3.2420909, -1.4674385, 1.3955431
5: -8.2896528, -6.8105326, -8.3198881, -6.8043079, -1.2743971, 1.2890959
6: -5.9173589, -4.0664520, -5.9275594, -4.0536022, -1.6241903, 1.6217155
7: -4.1273794, -2.8371000, -4.1388626, -2.8128054, -1.2576723, 1.2443004
8: -3.7275267, -2.3217211, -3.7589579, -2.3098330, -1.2873384, 1.2957923
9: -10.9146385, -9.1954670, -11.0403852, -9.1633463, -1.3905706, 1.3772875

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9362969, upper bound: 0.9499011
time: 3.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416704, upper bound: 0.9508000
time: 3.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8980446, -6.7183113, -1.8793178, 1.8556061
1: 2.0097160, 3.5735288, 1.9351292, 3.7009096, -1.3807387, 1.4516459
2: -5.4477758, -3.8951197, -5.4800196, -3.7877562, -1.3658545, 1.3823597
3: -10.1518946, -8.4342260, -10.1745787, -8.3877344, -1.2271800, 1.2154627
4: -4.7095294, -3.3394604, -4.7937937, -3.2352352, -1.4742942, 1.4543333
5: -8.2896528, -6.8105326, -8.3986349, -6.7877893, -1.3018701, 1.3415296
6: -5.9173589, -4.0664520, -5.9908152, -3.9284964, -1.6745813, 1.7163281
7: -4.1273794, -2.8371000, -4.2180338, -2.7867959, -1.3405836, 1.2962853
8: -3.7275267, -2.3217211, -3.7690148, -2.2899494, -1.3096180, 1.3046188
9: -10.9146385, -9.1954670, -11.1754637, -9.1288528, -1.5334439, 1.4269496

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9362956, upper bound: 0.9499021
time: 3.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416704, upper bound: 0.9508012
time: 3.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8731165, -6.7341380, -1.8698654, 1.8721223
1: 1.9655747, 3.6435614, 1.9800858, 3.6309133, -1.4342494, 1.4074497
2: -5.4579601, -3.8526134, -5.4694366, -3.8297918, -1.3754878, 1.3842790
3: -10.1621304, -8.4082613, -10.1643639, -8.4138031, -1.2146468, 1.2288365
4: -4.7673187, -3.3324428, -4.7350035, -3.2420909, -1.5252278, 1.4025607
5: -8.3700352, -6.7941961, -8.3198881, -6.8043079, -1.3371224, 1.3129983
6: -5.9793115, -3.9414463, -5.9275594, -4.0536022, -1.7162774, 1.6783936
7: -4.2065811, -2.8140025, -4.1388626, -2.8128054, -1.2931452, 1.3248601
8: -3.7371349, -2.3019562, -3.7589579, -2.3098330, -1.2980070, 1.3127131
9: -11.0496979, -9.1602507, -11.0403852, -9.1633463, -1.4587579, 1.5133045

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9362956, upper bound: 0.9722860
time: 3.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416704, upper bound: 0.9731849
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8981972, -6.7152972, -1.9672089, 1.9566002
1: 1.9655747, 3.6435614, 1.9334812, 3.7011271, -1.5298080, 1.5506928
2: -5.4579601, -3.8526134, -5.4810052, -3.7874231, -1.4293132, 1.4349101
3: -10.1621304, -8.4082613, -10.1747808, -8.3837280, -1.2832792, 1.2819928
4: -4.7673187, -3.3324428, -4.7960649, -3.2351470, -1.5321717, 1.4636221
5: -8.3700352, -6.7941961, -8.3994436, -6.7868481, -1.3840570, 1.3959275
6: -5.9793115, -3.9414463, -5.9932308, -3.9284909, -1.8677220, 1.8710089
7: -4.2065811, -2.8140025, -4.2181268, -2.7841980, -1.4223831, 1.4041243
8: -3.7371349, -2.3019562, -3.7703452, -2.2896948, -1.3238063, 1.3308690
9: -11.0496979, -9.1602507, -11.1756668, -9.1263037, -1.6751165, 1.6548489

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9362955, upper bound: 0.9840186
time: 3.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416703, upper bound: 0.9849374
time: 3.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.7869673, -6.7579565, -1.8404903, 1.8449876
1: 1.9800858, 3.6309133, 2.0097160, 3.5735288, -1.3718991, 1.3563395
2: -5.4694366, -3.8297918, -5.4477758, -3.8951197, -1.3452098, 1.3407414
3: -10.1643639, -8.4138031, -10.1518946, -8.4342260, -1.2044648, 1.2055798
4: -4.7350035, -3.2420909, -4.7095294, -3.3394604, -1.3955431, 1.4674385
5: -8.3198881, -6.8043079, -8.2896528, -6.8105326, -1.2890959, 1.2743971
6: -5.9275594, -4.0536022, -5.9173589, -4.0664520, -1.6217151, 1.6241902
7: -4.1388626, -2.8128054, -4.1273794, -2.8371000, -1.2443006, 1.2576723
8: -3.7589579, -2.3098330, -3.7275267, -2.3217211, -1.2957923, 1.2873385
9: -11.0403852, -9.1633463, -10.9146385, -9.1954670, -1.3772876, 1.3905706

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454249, upper bound: 0.9407716
time: 3.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507997, upper bound: 0.9416708
time: 3.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8123884, -6.7421303, -1.8721223, 1.8698654
1: 1.9800858, 3.6309133, 1.9655747, 3.6435614, -1.4074497, 1.4342494
2: -5.4694366, -3.8297918, -5.4579601, -3.8526134, -1.3842788, 1.3754878
3: -10.1643639, -8.4138031, -10.1621304, -8.4082613, -1.2288369, 1.2146467
4: -4.7350035, -3.2420909, -4.7673187, -3.3324428, -1.4025607, 1.5252278
5: -8.3198881, -6.8043079, -8.3700352, -6.7941961, -1.3129983, 1.3371226
6: -5.9275594, -4.0536022, -5.9793115, -3.9414463, -1.6783938, 1.7162777
7: -4.1388626, -2.8128054, -4.2065811, -2.8140025, -1.3248601, 1.2931449
8: -3.7589579, -2.3098330, -3.7371349, -2.3019562, -1.3127134, 1.2980069
9: -11.0403852, -9.1633463, -11.0496979, -9.1602507, -1.5133045, 1.4587579

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454249, upper bound: 0.9407728
time: 3.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507997, upper bound: 0.9416719
time: 4.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8980446, -6.7183113, -8.7869673, -6.7579565, -1.8556061, 1.8793180
1: 1.9351292, 3.7009096, 2.0097160, 3.5735288, -1.4516459, 1.3807392
2: -5.4800196, -3.7877562, -5.4477758, -3.8951197, -1.3823597, 1.3658543
3: -10.1745787, -8.3877344, -10.1518946, -8.4342260, -1.2154626, 1.2271799
4: -4.7937937, -3.2352352, -4.7095294, -3.3394604, -1.4543333, 1.4742942
5: -8.3986349, -6.7877893, -8.2896528, -6.8105326, -1.3415296, 1.3018702
6: -5.9908152, -3.9284964, -5.9173589, -4.0664520, -1.7163281, 1.6745808
7: -4.2180338, -2.7867959, -4.1273794, -2.8371000, -1.2962854, 1.3405836
8: -3.7690148, -2.2899494, -3.7275267, -2.3217211, -1.3046193, 1.3096182
9: -11.1754637, -9.1288528, -10.9146385, -9.1954670, -1.4269497, 1.5334436

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454249, upper bound: 0.9631622
time: 3.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507997, upper bound: 0.9640612
time: 3.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8123884, -6.7421303, -1.9566002, 1.9672091
1: 1.9334812, 3.7011271, 1.9655747, 3.6435614, -1.5506928, 1.5298080
2: -5.4810052, -3.7874231, -5.4579601, -3.8526134, -1.4349101, 1.4293127
3: -10.1747808, -8.3837280, -10.1621304, -8.4082613, -1.2819929, 1.2832791
4: -4.7960649, -3.2351470, -4.7673187, -3.3324428, -1.4636221, 1.5321717
5: -8.3994436, -6.7868481, -8.3700352, -6.7941961, -1.3959277, 1.3840570
6: -5.9932308, -3.9284909, -5.9793115, -3.9414463, -1.8710089, 1.8677220
7: -4.2181268, -2.7841980, -4.2065811, -2.8140025, -1.4041243, 1.4223831
8: -3.7703452, -2.2896948, -3.7371349, -2.3019562, -1.3308687, 1.3238063
9: -11.1756668, -9.1263037, -11.0496979, -9.1602507, -1.6548491, 1.6751163

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454247, upper bound: 0.9749020
time: 3.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507995, upper bound: 0.9758168
time: 4.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8731165, -6.7341380, -1.8653698, 1.8653700
1: 1.9800858, 3.6309133, 1.9800858, 3.6309133, -1.3847399, 1.3847396
2: -5.4694366, -3.8297918, -5.4694366, -3.8297918, -1.3633099, 1.3633099
3: -10.1643639, -8.4138031, -10.1643639, -8.4138031, -1.2166085, 1.2166088
4: -4.7350035, -3.2420909, -4.7350035, -3.2420909, -1.4929125, 1.4929125
5: -8.3198881, -6.8043079, -8.3198881, -6.8043079, -1.2975898, 1.2975898
6: -5.9275594, -4.0536022, -5.9275594, -4.0536022, -1.6430702, 1.6430696
7: -4.1388626, -2.8128054, -4.1388626, -2.8128054, -1.2636733, 1.2636729
8: -3.7589579, -2.3098330, -3.7589579, -2.3098330, -1.3070773, 1.3070774
9: -11.0403852, -9.1633463, -11.0403852, -9.1633463, -1.4160433, 1.4160433

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454205, upper bound: 0.9407716
time: 3.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507953, upper bound: 0.9416708
time: 3.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8981972, -6.7152972, -1.8999329, 1.8806412
1: 1.9800858, 3.6309133, 1.9334812, 3.7011271, -1.4094596, 1.4649146
2: -5.4694366, -3.8297918, -5.4810052, -3.7874231, -1.3897538, 1.3986037
3: -10.1643639, -8.4138031, -10.1747808, -8.3837280, -1.2415655, 1.2261428
4: -4.7350035, -3.2420909, -4.7960649, -3.2351470, -1.4998565, 1.5539739
5: -8.3198881, -6.8043079, -8.3994436, -6.7868481, -1.3222864, 1.3519796
6: -5.9275594, -4.0536022, -5.9932308, -3.9284909, -1.6876962, 1.7300591
7: -4.1388626, -2.8128054, -4.2181268, -2.7841980, -1.3546646, 1.3050520
8: -3.7589579, -2.3098330, -3.7703452, -2.2896948, -1.3292420, 1.3200587
9: -11.0403852, -9.1633463, -11.1756668, -9.1263037, -1.5471866, 1.4660139

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454205, upper bound: 0.9407727
time: 3.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507953, upper bound: 0.9416719
time: 4.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8731165, -6.7341380, -1.8806415, 1.8999329
1: 1.9334812, 3.7011271, 1.9800858, 3.6309133, -1.4649143, 1.4094595
2: -5.4810052, -3.7874231, -5.4694366, -3.8297918, -1.3986039, 1.3897541
3: -10.1747808, -8.3837280, -10.1643639, -8.4138031, -1.2261429, 1.2415655
4: -4.7960649, -3.2351470, -4.7350035, -3.2420909, -1.5539739, 1.4998565
5: -8.3994436, -6.7868481, -8.3198881, -6.8043079, -1.3519797, 1.3222864
6: -5.9932308, -3.9284909, -5.9275594, -4.0536022, -1.7300591, 1.6876962
7: -4.2181268, -2.7841980, -4.1388626, -2.8128054, -1.3050518, 1.3546646
8: -3.7703452, -2.2896948, -3.7589579, -2.3098330, -1.3200589, 1.3292420
9: -11.1756668, -9.1263037, -11.0403852, -9.1633463, -1.4660139, 1.5471867

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454205, upper bound: 0.9631622
time: 3.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454204, upper bound: 0.9640623
time: 3.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8981972, -6.7152972, -1.9836760, 1.9836762
1: 1.9334812, 3.7011271, 1.9334812, 3.7011271, -1.5604730, 1.5604730
2: -5.4810052, -3.7874231, -5.4810052, -3.7874231, -1.4524288, 1.4524286
3: -10.1747808, -8.3837280, -10.1747808, -8.3837280, -1.2964683, 1.2964683
4: -4.7960649, -3.2351470, -4.7960649, -3.2351470, -1.5609179, 1.5609179
5: -8.3994436, -6.7868481, -8.3994436, -6.7868481, -1.4055998, 1.4055998
6: -5.9932308, -3.9284909, -5.9932308, -3.9284909, -1.8819368, 1.8819369
7: -4.2181268, -2.7841980, -4.2181268, -2.7841980, -1.4339287, 1.4339287
8: -3.7703452, -2.2896948, -3.7703452, -2.2896948, -1.3456550, 1.3456550
9: -11.1756668, -9.1263037, -11.1756668, -9.1263037, -1.6887312, 1.6887312

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9454203, upper bound: 0.9631627
time: 3.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507952, upper bound: 0.9640617
time: 3.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.60 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9363093, upper bound: 0.9407854
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9416846
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9363093, upper bound: 0.9407850
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9416857
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9363093, upper bound: 0.9631811
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9640801
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9363107, upper bound: 0.9749162
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416842, upper bound: 0.9758346
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9362969, upper bound: 0.9499011
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416704, upper bound: 0.9508000
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9362956, upper bound: 0.9499021
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416704, upper bound: 0.9508012
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9362956, upper bound: 0.9722860
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416704, upper bound: 0.9731849
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9362955, upper bound: 0.9840186
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9416703, upper bound: 0.9849374
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454249, upper bound: 0.9407716
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507997, upper bound: 0.9416708
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454249, upper bound: 0.9407728
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507997, upper bound: 0.9416719
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454249, upper bound: 0.9631622
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507997, upper bound: 0.9640612
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454247, upper bound: 0.9749020
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507995, upper bound: 0.9758168
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454205, upper bound: 0.9407716
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507953, upper bound: 0.9416708
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454205, upper bound: 0.9407727
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507953, upper bound: 0.9416719
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454205, upper bound: 0.9631622
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454204, upper bound: 0.9640623
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9454203, upper bound: 0.9631627
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.60
Output dim: 1, lower bound: -0.9507952, upper bound: 0.9640617

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7812405, -6.7654772, -8.7856703, -6.7596965, -1.8080583, 1.8088267
1: 2.0216389, 3.5678782, 2.0124607, 3.5722408, -1.3263531, 1.3312835
2: -5.4225674, -3.8992851, -5.4419823, -3.8960648, -1.2957995, 1.3114653
3: -10.1476316, -8.4416676, -10.1509228, -8.4359503, -1.1828023, 1.1815877
4: -4.7011065, -3.3426561, -4.7075901, -3.3401875, -1.3609190, 1.3649340
5: -8.2864246, -6.8191590, -8.2889175, -6.8125257, -1.2589545, 1.2557580
6: -5.9099426, -4.0831575, -5.9156713, -4.0703020, -1.5961728, 1.5919137
7: -4.1140456, -2.8418369, -4.1243091, -2.8381817, -1.2149973, 1.2173122
8: -3.7169852, -2.3340378, -3.7249408, -2.3245530, -1.2496703, 1.2526982
9: -10.9085102, -9.2069473, -10.9132462, -9.1981125, -1.3416067, 1.3387436

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9362935, upper bound: 0.9373654
time: 3.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9362926, upper bound: 0.9407674
time: 3.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.7925138, -6.7545271, -8.7869644, -6.7579603, -1.8220103, 1.8260684
1: 2.0053992, 3.5811214, 2.0097208, 3.5735273, -1.3463602, 1.3510268
2: -5.4516602, -3.8736844, -5.4477634, -3.8951206, -1.3232121, 1.3394487
3: -10.1547756, -8.4304695, -10.1518936, -8.4342270, -1.1939087, 1.1977478
4: -4.7126689, -3.3338997, -4.7095256, -3.3394608, -1.3732080, 1.3756258
5: -8.2975283, -6.8085432, -8.2896509, -6.8105359, -1.2741647, 1.2680149
6: -5.9344378, -4.0637441, -5.9173565, -4.0664573, -1.6265244, 1.6116014
7: -4.1305499, -2.8258536, -4.1273727, -2.8371007, -1.2343411, 1.2367853
8: -3.7411423, -2.3200998, -3.7275224, -2.3217249, -1.2800168, 1.2729384
9: -10.9246960, -9.1938839, -10.9146357, -9.1954699, -1.3624773, 1.3545444

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416672, upper bound: 0.9382644
time: 3.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416664, upper bound: 0.9416664
time: 3.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7812405, -6.7654772, -8.8110733, -6.7440629, -1.8402293, 1.8337035
1: 2.0216389, 3.5678782, 1.9684558, 3.6422782, -1.3620195, 1.4112111
2: -5.4225674, -3.8992851, -5.4520884, -3.8535604, -1.3347147, 1.3487043
3: -10.1476316, -8.4416676, -10.1611710, -8.4101973, -1.2065310, 1.1925756
4: -4.7011065, -3.3426561, -4.7651353, -3.3331671, -1.3679395, 1.4224792
5: -8.2864246, -6.8191590, -8.3692636, -6.7962556, -1.2862258, 1.3180826
6: -5.9099426, -4.0831575, -5.9774890, -3.9452951, -1.6516075, 1.6843612
7: -4.1140456, -2.8418369, -4.2035227, -2.8151829, -1.2988627, 1.2679961
8: -3.7169852, -2.3340378, -3.7344227, -2.3047848, -1.2718909, 1.2623801
9: -10.9085102, -9.2069473, -11.0483179, -9.1630592, -1.4875436, 1.4066226

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9586946, upper bound: 0.9373648
time: 3.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9586951, upper bound: 0.9407667
time: 3.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.7925138, -6.7545271, -8.8123770, -6.7423162, -1.8541689, 1.8509359
1: 2.0053992, 3.5811214, 1.9656792, 3.6435461, -1.3820908, 1.4310005
2: -5.4516602, -3.8736844, -5.4578886, -3.8526356, -1.3621256, 1.3741241
3: -10.1547756, -8.4304695, -10.1621189, -8.4084930, -1.2177938, 1.2087302
4: -4.7126689, -3.3338997, -4.7671795, -3.3324490, -1.3802199, 1.4332798
5: -8.2975283, -6.8085432, -8.3699837, -6.7942557, -1.3014379, 1.3306837
6: -5.9344378, -4.0637441, -5.9791656, -3.9414527, -1.6710830, 1.7040823
7: -4.1305499, -2.8258536, -4.2065706, -2.8141115, -1.3164384, 1.2833531
8: -3.7411423, -2.3200998, -3.7370563, -2.3019753, -1.3022313, 1.2826494
9: -10.9246960, -9.1938839, -11.0496807, -9.1604071, -1.5083756, 1.4225941

Time for backsubstitution: 14.68 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.509117841720581
rel_dist={1: [-0.9849775715086642, 0.9849766529340247]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7554966
time: 3.59 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554946, upper bound: 0.7554956
time: 3.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.41
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7554966
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.41
Output dim: 1, lower bound: -0.7554946, upper bound: 0.7554956

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8134632, -6.7279902, -1.7136467, 1.7005167
1: 1.9655743, 3.6435642, 1.9534850, 3.6440954, -1.3565862, 1.3685148
2: -5.4579616, -3.8526118, -5.4696188, -3.8512685, -1.3111424, 1.3213866
3: -10.1621323, -8.4082584, -10.1650553, -8.4066534, -1.0299273, 1.0324852
4: -4.7673235, -3.3324428, -4.7845268, -3.3316565, -1.3887916, 1.4060545
5: -8.3700380, -6.7941961, -8.3735094, -6.7897739, -1.2160375, 1.2166901
6: -5.9793115, -3.9414420, -5.9832397, -3.9410968, -1.6582835, 1.6607938
7: -4.2065864, -2.8140006, -4.2095613, -2.8125353, -1.3572133, 1.3605845
8: -3.7371349, -2.3019557, -3.7387199, -2.2968702, -1.1581008, 1.1542064
9: -11.0497036, -9.1602478, -11.0502615, -9.1371479, -1.4785833, 1.4562726

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7464847
time: 3.62 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7554962
time: 3.56 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8134613, -6.7280145, -1.7325678, 1.7308986
1: 1.9334784, 3.7011304, 1.9534998, 3.6440954, -1.3872950, 1.3835762
2: -5.4810061, -3.7874212, -5.4696021, -3.8512707, -1.3348906, 1.3368154
3: -10.1747808, -8.3837261, -10.1650486, -8.4066563, -1.0489736, 1.0477846
4: -4.7960691, -3.2351460, -4.7845011, -3.3316588, -1.4249089, 1.4320772
5: -8.3994455, -6.7868471, -8.3735046, -6.7897778, -1.2356710, 1.2292193
6: -5.9932337, -3.9284849, -5.9832325, -3.9410970, -1.6743312, 1.6732727
7: -4.2181315, -2.7841961, -4.2095566, -2.8125374, -1.3900738, 1.3868465
8: -3.7703476, -2.2896943, -3.7387185, -2.2968788, -1.1801157, 1.1796849
9: -11.1756716, -9.1263027, -11.0502615, -9.1371822, -1.4992070, 1.4909601

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554972, upper bound: 0.7464846
time: 3.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554972, upper bound: 0.7554963
time: 3.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.91 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7464847
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7554962
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 1, lower bound: -0.7554972, upper bound: 0.7464846
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.91
Output dim: 1, lower bound: -0.7554972, upper bound: 0.7554963

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8123894, -6.7421303, -1.6993175, 1.6993179
1: 1.9655743, 3.6435642, 1.9655743, 3.6435642, -1.3558514, 1.3558515
2: -5.4579616, -3.8526118, -5.4579616, -3.8526118, -1.3098867, 1.3098867
3: -10.1621323, -8.4082584, -10.1621323, -8.4082584, -1.0280020, 1.0280020
4: -4.7673235, -3.3324428, -4.7673235, -3.3324428, -1.3879645, 1.3879645
5: -8.3700380, -6.7941961, -8.3700380, -6.7941961, -1.2110579, 1.2110579
6: -5.9793115, -3.9414420, -5.9793115, -3.9414420, -1.6576843, 1.6576838
7: -4.2065864, -2.8140006, -4.2065864, -2.8140006, -1.3541980, 1.3541980
8: -3.7371349, -2.3019557, -3.7371349, -2.3019557, -1.1521431, 1.1521432
9: -11.0497036, -9.1602478, -11.0497036, -9.1602478, -1.4554930, 1.4554925

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7326977, upper bound: 0.7192825
time: 3.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464815, upper bound: 0.7464872
time: 3.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8982000, -6.7152944, -1.7256117, 1.7182086
1: 1.9655743, 3.6435642, 1.9334784, 3.7011304, -1.3708599, 1.3860643
2: -5.4579616, -3.8526118, -5.4810061, -3.7874212, -1.3252661, 1.3329613
3: -10.1621323, -8.4082584, -10.1747808, -8.3837261, -1.0433216, 1.0397539
4: -4.7673235, -3.3324428, -4.7960691, -3.2351460, -1.4139798, 1.4166584
5: -8.3700380, -6.7941961, -8.3994455, -6.7868471, -1.2186029, 1.2306881
6: -5.9793115, -3.9414420, -5.9932337, -3.9284849, -1.6701684, 1.6730185
7: -4.2065864, -2.8140006, -4.2181315, -2.7841961, -1.3803627, 1.3659236
8: -3.7371349, -2.3019557, -3.7703476, -2.2896943, -1.1681209, 1.1741309
9: -11.0497036, -9.1602478, -11.1756716, -9.1263027, -1.4884071, 1.4761274

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7326977, upper bound: 0.7283518
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464815, upper bound: 0.7554917
time: 3.66 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8123894, -6.7421303, -1.7182083, 1.7256117
1: 1.9334784, 3.7011304, 1.9655743, 3.6435642, -1.3860643, 1.3708601
2: -5.4810061, -3.7874212, -5.4579616, -3.8526118, -1.3329613, 1.3252661
3: -10.1747808, -8.3837261, -10.1621323, -8.4082584, -1.0397537, 1.0433215
4: -4.7960691, -3.2351460, -4.7673235, -3.3324428, -1.4166584, 1.4139798
5: -8.3994455, -6.7868471, -8.3700380, -6.7941961, -1.2306879, 1.2186029
6: -5.9932337, -3.9284849, -5.9793115, -3.9414420, -1.6730185, 1.6701684
7: -4.2181315, -2.7841961, -4.2065864, -2.8140006, -1.3659234, 1.3803627
8: -3.7703476, -2.2896943, -3.7371349, -2.3019557, -1.1741307, 1.1681205
9: -11.1756716, -9.1263027, -11.0497036, -9.1602478, -1.4761274, 1.4884069

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7417947, upper bound: 0.7192769
time: 3.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554912, upper bound: 0.7464812
time: 3.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8982000, -6.7152944, -1.7435222, 1.7435226
1: 1.9334784, 3.7011304, 1.9334784, 3.7011304, -1.4011781, 1.4011779
2: -5.4810061, -3.7874212, -5.4810061, -3.7874212, -1.3482206, 1.3482206
3: -10.1747808, -8.3837261, -10.1747808, -8.3837261, -1.0564079, 1.0564080
4: -4.7960691, -3.2351460, -4.7960691, -3.2351460, -1.4369006, 1.4369004
5: -8.3994455, -6.7868471, -8.3994455, -6.7868471, -1.2369454, 1.2369454
6: -5.9932337, -3.9284849, -5.9932337, -3.9284849, -1.6835573, 1.6835576
7: -4.2181315, -2.7841961, -4.2181315, -2.7841961, -1.3969519, 1.3969519
8: -3.7703476, -2.2896943, -3.7703476, -2.2896943, -1.1864640, 1.1864638
9: -11.1756716, -9.1263027, -11.1756716, -9.1263027, -1.5045195, 1.5045197

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7417958, upper bound: 0.7192770
time: 3.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554922, upper bound: 0.7465566
time: 3.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7326977, upper bound: 0.7192825
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7464815, upper bound: 0.7464872
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7326977, upper bound: 0.7283518
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7464815, upper bound: 0.7554917
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7417947, upper bound: 0.7192769
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7554912, upper bound: 0.7464812
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7417958, upper bound: 0.7192770
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.02
Output dim: 1, lower bound: -0.7554922, upper bound: 0.7465566

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8045912, -6.7483349, -1.6699896, 1.6486118
1: 2.0097160, 3.5735288, 1.9789276, 3.6228161, -1.2235863, 1.2716548
2: -5.4477758, -3.8951197, -5.4538574, -3.8657565, -1.2380564, 1.2608337
3: -10.1518946, -8.4342260, -10.1588840, -8.4245224, -1.0046575, 0.9971318
4: -4.7095294, -3.3394604, -4.7445107, -3.3345873, -1.3309062, 1.3432941
5: -8.2896528, -6.8105326, -8.3446541, -6.7995744, -1.1205342, 1.1388822
6: -5.9173589, -4.0664520, -5.9605079, -3.9777658, -1.4720223, 1.5158690
7: -4.1273794, -2.8371000, -4.1834841, -2.8221979, -1.2702146, 1.1696944
8: -3.7275267, -2.3217211, -3.7323351, -2.3082156, -1.1348177, 1.1237307
9: -10.9146385, -9.1954670, -11.0102053, -9.1719627, -1.3181441, 1.2099370

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7192899
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7192904
time: 5.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8123894, -6.7421303, -1.6993146, 1.7522068
1: 1.9655747, 3.6435614, 1.9655743, 3.6435642, -1.3780348, 1.3558487
2: -5.4579601, -3.8526134, -5.4579616, -3.8526118, -1.3140175, 1.3098845
3: -10.1621304, -8.4082613, -10.1621323, -8.4082584, -1.0271761, 1.0630952
4: -4.7673187, -3.3324428, -4.7673235, -3.3324428, -1.4265440, 1.3853176
5: -8.3700352, -6.7941961, -8.3700380, -6.7941961, -1.2048206, 1.2110573
6: -5.9793115, -3.9414463, -5.9793115, -3.9414420, -1.6806338, 1.6576807
7: -4.2065811, -2.8140025, -4.2065864, -2.8140006, -1.3541942, 1.3925838
8: -3.7371349, -2.3019562, -3.7371349, -2.3019557, -1.1542358, 1.1521425
9: -11.0496979, -9.1602507, -11.0497036, -9.1602478, -1.4554873, 1.4618431

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7327109
time: 3.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7327121
time: 4.89 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8904219, -6.7246461, -1.6955676, 1.6617835
1: 2.0097160, 3.5735288, 1.9487343, 3.6801958, -1.2252810, 1.3002841
2: -5.4477758, -3.8951197, -5.4758506, -3.8007462, -1.2467351, 1.2833281
3: -10.1518946, -8.4342260, -10.1713247, -8.4042835, -1.0171452, 1.0080279
4: -4.7095294, -3.3394604, -4.7707558, -3.2373271, -1.3567753, 1.3689203
5: -8.2896528, -6.8105326, -8.3738661, -6.7931910, -1.1272781, 1.1499369
6: -5.9173589, -4.0664520, -5.9715862, -3.9648428, -1.4784267, 1.5276024
7: -4.1273794, -2.8371000, -4.1949558, -2.7975378, -1.2948985, 1.1805751
8: -3.7275267, -2.3217211, -3.7639437, -2.2962346, -1.1496407, 1.1448107
9: -10.9146385, -9.1954670, -11.1359854, -9.1402721, -1.3507950, 1.2137940

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7283513
time: 3.68 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7283519
time: 5.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8982000, -6.7152944, -1.7256088, 1.7667007
1: 1.9655747, 3.6435614, 1.9334784, 3.7011304, -1.3859699, 1.3860614
2: -5.4579601, -3.8526134, -5.4810061, -3.7874212, -1.3270605, 1.3329592
3: -10.1621304, -8.4082613, -10.1747808, -8.3837261, -1.0425239, 1.0749407
4: -4.7673187, -3.3324428, -4.7960691, -3.2351460, -1.4497700, 1.4140112
5: -8.3700352, -6.7941961, -8.3994455, -6.7868471, -1.2123661, 1.2273524
6: -5.9793115, -3.9414463, -5.9932337, -3.9284849, -1.6892114, 1.6730151
7: -4.2065811, -2.8140025, -4.2181315, -2.7841961, -1.3803632, 1.4041290
8: -3.7371349, -2.3019562, -3.7703476, -2.2896943, -1.1702123, 1.1741302
9: -11.0496979, -9.1602507, -11.1756716, -9.1263027, -1.4884019, 1.4711421

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7417943
time: 3.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7417953
time: 4.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8047285, -6.7477493, -1.6886425, 1.6725209
1: 1.9800858, 3.6309133, 1.9773893, 3.6230354, -1.2522988, 1.2876134
2: -5.4694366, -3.8297918, -5.4548101, -3.8654499, -1.2619742, 1.2743442
3: -10.1643639, -8.4138031, -10.1590710, -8.4220133, -1.0165232, 1.0074017
4: -4.7350035, -3.2420909, -4.7462525, -3.3344994, -1.3553064, 1.3702981
5: -8.3198881, -6.8043079, -8.3454084, -6.7986641, -1.1398995, 1.1465189
6: -5.9275594, -4.0536022, -5.9627361, -3.9777613, -1.4821968, 1.5309516
7: -4.1388626, -2.8128054, -4.1835728, -2.8204596, -1.2823634, 1.1775553
8: -3.7589579, -2.3098330, -3.7334547, -2.3079948, -1.1519818, 1.1405759
9: -11.0403852, -9.1633463, -11.0104055, -9.1694965, -1.3379078, 1.2458975

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283519, upper bound: 0.7192775
time: 3.78 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283519, upper bound: 0.7192757
time: 4.08 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8123894, -6.7421303, -1.7182083, 1.7784913
1: 1.9334812, 3.7011271, 1.9655743, 3.6435642, -1.4082472, 1.3708596
2: -5.4810052, -3.7874231, -5.4579616, -3.8526118, -1.3370922, 1.3252652
3: -10.1747808, -8.3837280, -10.1621323, -8.4082584, -1.0390294, 1.0743692
4: -4.7960649, -3.2351470, -4.7673235, -3.3324428, -1.4536207, 1.4113808
5: -8.3994436, -6.7868481, -8.3700380, -6.7941961, -1.2244489, 1.2186019
6: -5.9932308, -3.9284909, -5.9793115, -3.9414420, -1.6959665, 1.6701651
7: -4.2181268, -2.7841980, -4.2065864, -2.8140006, -1.3659172, 1.4146438
8: -3.7703452, -2.2896948, -3.7371349, -2.3019557, -1.1753561, 1.1681191
9: -11.1756668, -9.1263037, -11.0497036, -9.1602478, -1.4761269, 1.4947577

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283518, upper bound: 0.7326966
time: 3.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283519, upper bound: 0.7326950
time: 4.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8906231, -6.7237158, -1.7135372, 1.6868665
1: 1.9800858, 3.6309133, 1.9463000, 3.6805296, -1.2541683, 1.3168166
2: -5.4694366, -3.8297918, -5.4773307, -3.8002727, -1.2708087, 1.2968583
3: -10.1643639, -8.4138031, -10.1716328, -8.3996754, -1.0307505, 1.0186651
4: -4.7350035, -3.2420909, -4.7735248, -3.2371917, -1.3752818, 1.3901691
5: -8.3198881, -6.8043079, -8.3749752, -6.7917705, -1.1443408, 1.1606780
6: -5.9275594, -4.0536022, -5.9751291, -3.9648333, -1.4915605, 1.5417385
7: -4.1388626, -2.8128054, -4.1950970, -2.7936354, -1.3129153, 1.1893816
8: -3.7589579, -2.3098330, -3.7658329, -2.2958851, -1.1663084, 1.1585822
9: -11.0403852, -9.1633463, -11.1362963, -9.1364498, -1.3668253, 1.2530167

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7192775
time: 3.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7192751
time: 4.04 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8982000, -6.7152944, -1.7435184, 1.7937768
1: 1.9334812, 3.7011271, 1.9334784, 3.7011304, -1.4166353, 1.4011776
2: -5.4810052, -3.7874231, -5.4810061, -3.7874212, -1.3501761, 1.3482196
3: -10.1747808, -8.3837280, -10.1747808, -8.3837261, -1.0557134, 1.0875585
4: -4.7960649, -3.2351470, -4.7960691, -3.2351460, -1.4738955, 1.4342942
5: -8.3994436, -6.7868481, -8.3994455, -6.7868471, -1.2307103, 1.2369448
6: -5.9932308, -3.9284909, -5.9932337, -3.9284849, -1.7050271, 1.6835538
7: -4.2181268, -2.7841980, -4.2181315, -2.7841961, -1.3969519, 1.4312322
8: -3.7703452, -2.2896948, -3.7703476, -2.2896943, -1.1885569, 1.1864629
9: -11.1756668, -9.1263037, -11.1756716, -9.1263027, -1.5045142, 1.5050242

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7326967
time: 3.88 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7465559
time: 4.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7192899
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7192904
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7327109
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192904, upper bound: 0.7327121
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7283513
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7283519
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7417943
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7192781, upper bound: 0.7417953
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283519, upper bound: 0.7192775
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283519, upper bound: 0.7192757
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283518, upper bound: 0.7326966
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283519, upper bound: 0.7326950
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7192775
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7192751
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7326967
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.80
Output dim: 1, lower bound: -0.7283515, upper bound: 0.7465559

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.7869673, -6.7579565, -1.6342249, 1.6342249
1: 2.0097160, 3.5735288, 2.0097160, 3.5735288, -1.1998980, 1.1998978
2: -5.4477758, -3.8951197, -5.4477758, -3.8951197, -1.2190526, 1.2190526
3: -10.1518946, -8.4342260, -10.1518946, -8.4342260, -0.9901766, 0.9901764
4: -4.7095294, -3.3394604, -4.7095294, -3.3394604, -1.3219891, 1.3219891
5: -8.2896528, -6.8105326, -8.2896528, -6.8105326, -1.0980053, 1.0980053
6: -5.9173589, -4.0664520, -5.9173589, -4.0664520, -1.4355197, 1.4355196
7: -4.1273794, -2.8371000, -4.1273794, -2.8371000, -1.1344130, 1.1344130
8: -3.7275267, -2.3217211, -3.7275267, -2.3217211, -1.1195662, 1.1195662
9: -10.9146385, -9.1954670, -10.9146385, -9.1954670, -1.1654980, 1.1654980

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7183471
time: 3.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192852, upper bound: 0.7192851
time: 3.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8118305, -6.7455721, -1.6677418, 1.6585336
1: 2.0097160, 3.5735288, 1.9710317, 3.6428018, -1.2237902, 1.2777014
2: -5.4477758, -3.8951197, -5.4546261, -3.8537545, -1.2462206, 1.2611852
3: -10.1518946, -8.4342260, -10.1614876, -8.4205389, -1.0060825, 0.9995983
4: -4.7095294, -3.3394604, -4.7598228, -3.3327475, -1.3287923, 1.3651175
5: -8.2896528, -6.8105326, -8.3671227, -6.7973547, -1.1224887, 1.1478395
6: -5.9173589, -4.0664520, -5.9713421, -3.9414659, -1.4728339, 1.5216012
7: -4.1273794, -2.8371000, -4.2062769, -2.8200872, -1.2691803, 1.1715479
8: -3.7275267, -2.3217211, -3.7331705, -2.3027992, -1.1408979, 1.1248381
9: -10.9146385, -9.1954670, -11.0490026, -9.1688643, -1.3206818, 1.2119384

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7183451
time: 4.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7192856
time: 5.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118305, -6.7455721, -8.7869673, -6.7579565, -1.6585336, 1.6677418
1: 1.9710317, 3.6428018, 2.0097160, 3.5735288, -1.2777016, 1.2237902
2: -5.4546261, -3.8537545, -5.4477758, -3.8951197, -1.2611852, 1.2462208
3: -10.1614876, -8.4205389, -10.1518946, -8.4342260, -0.9995981, 1.0060823
4: -4.7598228, -3.3327475, -4.7095294, -3.3394604, -1.3651175, 1.3287923
5: -8.3671227, -6.7973547, -8.2896528, -6.8105326, -1.1478395, 1.1224887
6: -5.9713421, -3.9414659, -5.9173589, -4.0664520, -1.5216012, 1.4728341
7: -4.2062769, -2.8200872, -4.1273794, -2.8371000, -1.1715481, 1.2691803
8: -3.7331705, -2.3027992, -3.7275267, -2.3217211, -1.1248381, 1.1408980
9: -11.0490026, -9.1688643, -10.9146385, -9.1954670, -1.2119384, 1.3206815

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7314563
time: 3.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7327044
time: 4.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8123884, -6.7421303, -1.7522039, 1.7522044
1: 1.9655747, 3.6435614, 1.9655747, 3.6435614, -1.3780320, 1.3780320
2: -5.4579601, -3.8526134, -5.4579601, -3.8526134, -1.3140152, 1.3140152
3: -10.1621304, -8.4082613, -10.1621304, -8.4082613, -1.0630937, 1.0630934
4: -4.7673187, -3.3324428, -4.7673187, -3.3324428, -1.4265418, 1.4265416
5: -8.3700352, -6.7941961, -8.3700352, -6.7941961, -1.2048202, 1.2048200
6: -5.9793115, -3.9414463, -5.9793115, -3.9414463, -1.6806307, 1.6806307
7: -4.2065811, -2.8140025, -4.2065811, -2.8140025, -1.3925786, 1.3925786
8: -3.7371349, -2.3019562, -3.7371349, -2.3019562, -1.1542349, 1.1542348
9: -11.0496979, -9.1602507, -11.0496979, -9.1602507, -1.4618380, 1.4618380

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7314556
time: 4.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7327049
time: 8.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8731165, -6.7341380, -1.6580076, 1.6520925
1: 2.0097160, 3.5735288, 1.9800858, 3.6309133, -1.2115364, 1.2284354
2: -5.4477758, -3.8951197, -5.4694366, -3.8297918, -1.2326956, 1.2416106
3: -10.1518946, -8.4342260, -10.1643639, -8.4138031, -1.0013149, 1.0010636
4: -4.7095294, -3.3394604, -4.7350035, -3.2420909, -1.3478575, 1.3462622
5: -8.2896528, -6.8105326, -8.3198881, -6.8043079, -1.1046586, 1.1169254
6: -5.9173589, -4.0664520, -5.9275594, -4.0536022, -1.4483185, 1.4458435
7: -4.1273794, -2.8371000, -4.1388626, -2.8128054, -1.1587982, 1.1454260
8: -3.7275267, -2.3217211, -3.7589579, -2.3098330, -1.1343399, 1.1403723
9: -10.9146385, -9.1954670, -11.0403852, -9.1633463, -1.2010248, 1.1844052

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7274440
time: 3.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192728, upper bound: 0.7283466
time: 3.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.7869673, -6.7579565, -8.8975296, -6.7219181, -1.6935449, 1.6645935
1: 2.0097160, 3.5735288, 1.9406109, 3.7001848, -1.2255020, 1.3065429
2: -5.4477758, -3.8951197, -5.4767432, -3.7888451, -1.2503352, 1.2837856
3: -10.1518946, -8.4342260, -10.1739016, -8.4004984, -1.0174375, 1.0104835
4: -4.7095294, -3.3394604, -4.7862811, -3.2355318, -1.3529005, 1.3909955
5: -8.2896528, -6.8105326, -8.3959236, -6.7909117, -1.1292779, 1.1589963
6: -5.9173589, -4.0664520, -5.9827747, -3.9285192, -1.4791546, 1.5338202
7: -4.1273794, -2.8371000, -4.2177248, -2.7954268, -1.2869654, 1.1823978
8: -3.7275267, -2.3217211, -3.7648306, -2.2908001, -1.1557174, 1.1449432
9: -10.9146385, -9.1954670, -11.1747866, -9.1373081, -1.3531578, 1.2157966

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7274414
time: 5.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192728, upper bound: 0.7283471
time: 4.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8120794, -6.7446527, -8.8731165, -6.7341380, -1.6825681, 1.6864512
1: 1.9685984, 3.6431417, 1.9800858, 3.6309133, -1.2886934, 1.2526786
2: -5.4561129, -3.8532443, -5.4694366, -3.8297918, -1.2724528, 1.2703550
3: -10.1617756, -8.4153814, -10.1643639, -8.4138031, -1.0092559, 1.0211709
4: -4.7631578, -3.3326111, -4.7350035, -3.2420909, -1.3768849, 1.3536797
5: -8.3684273, -6.7959471, -8.3198881, -6.8043079, -1.1560998, 1.1385450
6: -5.9748964, -3.9414573, -5.9275594, -4.0536022, -1.5356803, 1.4830134
7: -4.2064128, -2.8173740, -4.1388626, -2.8128054, -1.1794651, 1.2815182
8: -3.7349219, -2.3024230, -3.7589579, -2.3098330, -1.1423473, 1.1557109
9: -11.0493126, -9.1650257, -11.0403852, -9.1633463, -1.2480752, 1.3286884

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7405522
time: 3.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192708, upper bound: 0.7417877
time: 4.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8981972, -6.7152972, -1.7784891, 1.7666998
1: 1.9655747, 3.6435614, 1.9334812, 3.7011271, -1.3859696, 1.4082439
2: -5.4579601, -3.8526134, -5.4810052, -3.7874231, -1.3270593, 1.3370895
3: -10.1621304, -8.4082613, -10.1747808, -8.3837280, -1.0743690, 1.0749383
4: -4.7673187, -3.3324428, -4.7960649, -3.2351470, -1.4497695, 1.4536185
5: -8.3700352, -6.7941961, -8.3994436, -6.7868481, -1.2123654, 1.2218034
6: -5.9793115, -3.9414463, -5.9932308, -3.9284909, -1.6892109, 1.6959636
7: -4.2065811, -2.8140025, -4.2181268, -2.7841980, -1.4146438, 1.4041243
8: -3.7371349, -2.3019562, -3.7703452, -2.2896948, -1.1702108, 1.1753554
9: -11.0496979, -9.1602507, -11.1756668, -9.1263037, -1.4947529, 1.4711416

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7541871
time: 5.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192708, upper bound: 0.7417899
time: 4.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.7869673, -6.7579565, -1.6520925, 1.6580076
1: 1.9800858, 3.6309133, 2.0097160, 3.5735288, -1.2284353, 1.2115359
2: -5.4694366, -3.8297918, -5.4477758, -3.8951197, -1.2416105, 1.2326958
3: -10.1643639, -8.4138031, -10.1518946, -8.4342260, -1.0010635, 1.0013150
4: -4.7350035, -3.2420909, -4.7095294, -3.3394604, -1.3462622, 1.3478575
5: -8.3198881, -6.8043079, -8.2896528, -6.8105326, -1.1169255, 1.1046586
6: -5.9275594, -4.0536022, -5.9173589, -4.0664520, -1.4458435, 1.4483184
7: -4.1388626, -2.8128054, -4.1273794, -2.8371000, -1.1454260, 1.1587980
8: -3.7589579, -2.3098330, -3.7275267, -2.3217211, -1.1403720, 1.1343400
9: -11.0403852, -9.1633463, -10.9146385, -9.1954670, -1.1844051, 1.2010248

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7183350
time: 3.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283466, upper bound: 0.7192728
time: 3.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8120794, -6.7446527, -1.6864514, 1.6825681
1: 1.9800858, 3.6309133, 1.9685984, 3.6431417, -1.2526786, 1.2886934
2: -5.4694366, -3.8297918, -5.4561129, -3.8532443, -1.2703547, 1.2724531
3: -10.1643639, -8.4138031, -10.1617756, -8.4153814, -1.0211709, 1.0092560
4: -4.7350035, -3.2420909, -4.7631578, -3.3326111, -1.3536797, 1.3768849
5: -8.3198881, -6.8043079, -8.3684273, -6.7959471, -1.1385450, 1.1560998
6: -5.9275594, -4.0536022, -5.9748964, -3.9414573, -1.4830132, 1.5356801
7: -4.1388626, -2.8128054, -4.2064128, -2.8173740, -1.2815180, 1.1794653
8: -3.7589579, -2.3098330, -3.7349219, -2.3024230, -1.1557109, 1.1423472
9: -11.0403852, -9.1633463, -11.0493126, -9.1650257, -1.3286884, 1.2480752

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7183329
time: 4.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283447, upper bound: 0.7192712
time: 4.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8975296, -6.7219181, -8.7869673, -6.7579565, -1.6645932, 1.6935449
1: 1.9406109, 3.7001848, 2.0097160, 3.5735288, -1.3065431, 1.2255023
2: -5.4767432, -3.7888451, -5.4477758, -3.8951197, -1.2837856, 1.2503352
3: -10.1739016, -8.4004984, -10.1518946, -8.4342260, -1.0104836, 1.0174375
4: -4.7862811, -3.2355318, -4.7095294, -3.3394604, -1.3909955, 1.3529003
5: -8.3959236, -6.7909117, -8.2896528, -6.8105326, -1.1589963, 1.1292781
6: -5.9827747, -3.9285192, -5.9173589, -4.0664520, -1.5338206, 1.4791543
7: -4.2177248, -2.7954268, -4.1273794, -2.8371000, -1.1823976, 1.2869654
8: -3.7648306, -2.2908001, -3.7275267, -2.3217211, -1.1449437, 1.1557173
9: -11.1747866, -9.1373081, -10.9146385, -9.1954670, -1.2157967, 1.3531575

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7314420
time: 3.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283466, upper bound: 0.7326901
time: 4.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8123884, -6.7421303, -1.7666998, 1.7784889
1: 1.9334812, 3.7011271, 1.9655747, 3.6435614, -1.4082439, 1.3859696
2: -5.4810052, -3.7874231, -5.4579601, -3.8526134, -1.3370895, 1.3270590
3: -10.1747808, -8.3837280, -10.1621304, -8.4082613, -1.0749383, 1.0743690
4: -4.7960649, -3.2351470, -4.7673187, -3.3324428, -1.4536185, 1.4497695
5: -8.3994436, -6.7868481, -8.3700352, -6.7941961, -1.2218037, 1.2123654
6: -5.9932308, -3.9284909, -5.9793115, -3.9414463, -1.6959634, 1.6892111
7: -4.2181268, -2.7841980, -4.2065811, -2.8140025, -1.4041243, 1.4146438
8: -3.7703452, -2.2896948, -3.7371349, -2.3019562, -1.1753554, 1.1702111
9: -11.1756668, -9.1263037, -11.0496979, -9.1602507, -1.4711416, 1.4947526

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7451224
time: 4.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283447, upper bound: 0.7464745
time: 4.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8731165, -6.7341380, -1.6746540, 1.6746542
1: 1.9800858, 3.6309133, 1.9800858, 3.6309133, -1.2399359, 1.2399364
2: -5.4694366, -3.8297918, -5.4694366, -3.8297918, -1.2552643, 1.2552643
3: -10.1643639, -8.4138031, -10.1643639, -8.4138031, -1.0120480, 1.0120480
4: -4.7350035, -3.2420909, -4.7350035, -3.2420909, -1.3658898, 1.3658900
5: -8.3198881, -6.8043079, -8.3198881, -6.8043079, -1.1238604, 1.1238604
6: -5.9275594, -4.0536022, -5.9275594, -4.0536022, -1.4652967, 1.4652966
7: -4.1388626, -2.8128054, -4.1388626, -2.8128054, -1.1629539, 1.1629535
8: -3.7589579, -2.3098330, -3.7589579, -2.3098330, -1.1507519, 1.1507522
9: -11.0403852, -9.1633463, -11.0403852, -9.1633463, -1.2207030, 1.2207032

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7183350
time: 3.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7192727
time: 4.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.8977642, -6.7209744, -1.7116504, 1.6897058
1: 1.9800858, 3.6309133, 1.9381242, 3.7005148, -1.2543912, 1.3171549
2: -5.4694366, -3.8297918, -5.4782290, -3.7883635, -1.2745051, 1.2948666
3: -10.1643639, -8.4138031, -10.1742096, -8.3949976, -1.0318685, 1.0203048
4: -4.7350035, -3.2420909, -4.7896814, -3.2353973, -1.3733354, 1.4038327
5: -8.3198881, -6.8043079, -8.3971567, -6.7894955, -1.1462917, 1.1699463
6: -5.9275594, -4.0536022, -5.9864240, -3.9285102, -1.4922757, 1.5479929
7: -4.1388626, -2.8128054, -4.2178650, -2.7915127, -1.3054285, 1.1912239
8: -3.7589579, -2.3098330, -3.7667265, -2.2904148, -1.1719792, 1.1596992
9: -11.0403852, -9.1633463, -11.1750956, -9.1334801, -1.3615377, 1.2550259

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7183335
time: 4.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7192708
time: 5.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8977642, -6.7209744, -8.8731165, -6.7341380, -1.6897058, 1.7116504
1: 1.9381242, 3.7005148, 1.9800858, 3.6309133, -1.3171549, 1.2543912
2: -5.4782290, -3.7883635, -5.4694366, -3.8297918, -1.2948670, 1.2745051
3: -10.1742096, -8.3949976, -10.1643639, -8.4138031, -1.0203049, 1.0318685
4: -4.7896814, -3.2353973, -4.7350035, -3.2420909, -1.4038329, 1.3733354
5: -8.3971567, -6.7894955, -8.3198881, -6.8043079, -1.1699464, 1.1462919
6: -5.9864240, -3.9285102, -5.9275594, -4.0536022, -1.5479932, 1.4922755
7: -4.2178650, -2.7915127, -4.1388626, -2.8128054, -1.1912241, 1.3054287
8: -3.7667265, -2.2904148, -3.7589579, -2.3098330, -1.1596991, 1.1719792
9: -11.1750956, -9.1334801, -11.0403852, -9.1633463, -1.2550259, 1.3615377

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7314420
time: 3.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7326899
time: 4.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8981972, -6.7152972, -8.8981972, -6.7152972, -1.7937760, 1.7937758
1: 1.9334812, 3.7011271, 1.9334812, 3.7011271, -1.4166350, 1.4166348
2: -5.4810052, -3.7874231, -5.4810052, -3.7874231, -1.3501749, 1.3501751
3: -10.1747808, -8.3837280, -10.1747808, -8.3837280, -1.0875580, 1.0875580
4: -4.7960649, -3.2351470, -4.7960649, -3.2351470, -1.4738939, 1.4738936
5: -8.3994436, -6.7868481, -8.3994436, -6.7868481, -1.2307096, 1.2307096
6: -5.9932308, -3.9284909, -5.9932308, -3.9284909, -1.7050271, 1.7050273
7: -4.2181268, -2.7841980, -4.2181268, -2.7841980, -1.4312325, 1.4312325
8: -3.7703452, -2.2896948, -3.7703452, -2.2896948, -1.1885557, 1.1885556
9: -11.1756668, -9.1263037, -11.1756668, -9.1263037, -1.5050237, 1.5050237

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7451816
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283462, upper bound: 0.7326895
time: 4.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.14 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7183471
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192852, upper bound: 0.7192851
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7183451
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7192856
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7314563
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7327044
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7314556
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7327049
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7274440
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192728, upper bound: 0.7283466
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7274414
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192728, upper bound: 0.7283471
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7405522
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192708, upper bound: 0.7417877
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7541871
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7192708, upper bound: 0.7417899
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7183350
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283466, upper bound: 0.7192728
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7183329
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283447, upper bound: 0.7192712
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7314420
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283466, upper bound: 0.7326901
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7451224
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283447, upper bound: 0.7464745
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7183350
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7192727
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7183335
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7192708
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7314420
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7326899
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7451816
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.14
Output dim: 1, lower bound: -0.7283462, upper bound: 0.7326895

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7812405, -6.7654772, -8.7842255, -6.7616034, -1.6185861, 1.6190829
1: 2.0216389, 3.5678782, 2.0154829, 3.5708132, -1.1807141, 1.1840060
2: -5.4225674, -3.8992851, -5.4355974, -3.8971150, -1.1908669, 1.2013667
3: -10.1476316, -8.4416676, -10.1498470, -8.4378414, -0.9768714, 0.9760320
4: -4.7011065, -3.3426561, -4.7054543, -3.3409958, -1.3051727, 1.3070095
5: -8.2864246, -6.8191590, -8.2881012, -6.8147159, -1.0865610, 1.0843958
6: -5.9099426, -4.0831575, -5.9138002, -4.0745382, -1.4160392, 1.4131571
7: -4.1140456, -2.8418369, -4.1209288, -2.8393805, -1.1129665, 1.1145350
8: -3.7169852, -2.3340378, -3.7222395, -2.3276734, -1.0927808, 1.0947804
9: -10.9085102, -9.2069473, -10.9116993, -9.2010231, -1.1485271, 1.1466010

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7154640, upper bound: 0.7156267
time: 3.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155218, upper bound: 0.7183434
time: 3.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.7925138, -6.7545271, -8.7869644, -6.7579613, -1.6350286, 1.6380775
1: 2.0053992, 3.5811214, 2.0097256, 3.5735264, -1.2018828, 1.2075586
2: -5.4516602, -3.8736844, -5.4477539, -3.8951211, -1.2167354, 1.2314005
3: -10.1547756, -8.4304695, -10.1518917, -8.4342289, -0.9905059, 0.9938526
4: -4.7126689, -3.3338997, -4.7095222, -3.3394611, -1.3243678, 1.3237004
5: -8.2975283, -6.8085432, -8.2896490, -6.8105402, -1.1044230, 1.0969419
6: -5.9344378, -4.0637441, -5.9173555, -4.0664616, -1.4502993, 1.4325259
7: -4.1305499, -2.8258536, -4.1273675, -2.8371017, -1.1323857, 1.1379066
8: -3.7411423, -2.3200998, -3.7275229, -2.3217297, -1.1270161, 1.1166497
9: -10.9246960, -9.1938839, -10.9146328, -9.1954727, -1.1729289, 1.1632607

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192092, upper bound: 0.7165461
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192797, upper bound: 0.7192811
time: 3.98 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.66 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.66
Output dim: 1, lower bound: -0.7154640, upper bound: 0.7156267
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.66
Output dim: 1, lower bound: -0.7155218, upper bound: 0.7183434
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.66
Output dim: 1, lower bound: -0.7192092, upper bound: 0.7165461
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.66
Output dim: 1, lower bound: -0.7192797, upper bound: 0.7192811
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7183451
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7192856
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7314563
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7327044
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155252, upper bound: 0.7314556
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192832, upper bound: 0.7327049
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7274440
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192728, upper bound: 0.7283466
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7274414
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192728, upper bound: 0.7283471
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7405522
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192708, upper bound: 0.7417877
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7155130, upper bound: 0.7541871
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7192708, upper bound: 0.7417899
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7183350
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283466, upper bound: 0.7192728
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7183329
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283447, upper bound: 0.7192712
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7314420
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283466, upper bound: 0.7326901
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246212, upper bound: 0.7451224
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283447, upper bound: 0.7464745
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7183350
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7192727
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7183335
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7192708
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7314420
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283442, upper bound: 0.7326899
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7246208, upper bound: 0.7451816
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.66
Output dim: 1, lower bound: -0.7283462, upper bound: 0.7326895
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3692495822906494
rel_dist={1: [-0.7555267226325175, 0.7555256824713692]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6660908, upper bound: 0.6733572
time: 3.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734502, upper bound: 0.6734524
time: 3.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.08
Output dim: 1, lower bound: -0.6660908, upper bound: 0.6733572
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.08
Output dim: 1, lower bound: -0.6734502, upper bound: 0.6734524

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8132877, -6.7303009, -1.6504507, 1.6394632
1: 1.9655743, 3.6435642, 1.9554653, 3.6440086, -1.3098434, 1.3198190
2: -5.4579616, -3.8526118, -5.4677148, -3.8514876, -1.2784901, 1.2870603
3: -10.1621323, -8.4082584, -10.1645718, -8.4069157, -0.9621580, 0.9643015
4: -4.7673235, -3.3324428, -4.7817125, -3.3317838, -1.3402495, 1.3546939
5: -8.3700380, -6.7941961, -8.3729439, -6.7904968, -1.1583042, 1.1588397
6: -5.9793115, -3.9414420, -5.9825916, -3.9411530, -1.6007273, 1.6028214
7: -4.2065864, -2.8140006, -4.2090664, -2.8127747, -1.3246415, 1.3274555
8: -3.7371349, -2.3019557, -3.7384624, -2.2977057, -1.1060107, 1.1027495
9: -11.0497036, -9.1602478, -11.0501690, -9.1409216, -1.4149337, 1.3962684

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6660908, upper bound: 0.6660904
time: 3.45 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6660908, upper bound: 0.6733573
time: 3.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8134594, -6.7280216, -1.6713223, 1.6664774
1: 1.9334784, 3.7011304, 1.9535041, 3.6440935, -1.3390955, 1.3364909
2: -5.4810061, -3.7874212, -5.4695988, -3.8512712, -1.3003247, 1.3028908
3: -10.1747808, -8.3837261, -10.1650486, -8.4066563, -0.9801023, 0.9797162
4: -4.7960691, -3.2351460, -4.7844954, -3.3316591, -1.3721960, 1.3830163
5: -8.3994455, -6.7868471, -8.3735027, -6.7897782, -1.1779418, 1.1710796
6: -5.9932337, -3.9284849, -5.9832315, -3.9410977, -1.6167414, 1.6158134
7: -4.2181315, -2.7841961, -4.2095542, -2.8125379, -1.3551335, 1.3532333
8: -3.7703476, -2.2896943, -3.7387180, -2.2968798, -1.1283588, 1.1273992
9: -11.1756716, -9.1263027, -11.0502615, -9.1371880, -1.4382176, 1.4255655

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6486312, upper bound: 0.6589527
time: 3.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734441, upper bound: 0.6734473
time: 3.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.99 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 21.99
Output dim: 1, lower bound: -0.6660908, upper bound: 0.6660904
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.99
Output dim: 1, lower bound: -0.6660908, upper bound: 0.6733573
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 21.99
Output dim: 1, lower bound: -0.6486312, upper bound: 0.6589527
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.99
Output dim: 1, lower bound: -0.6734441, upper bound: 0.6734473

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8982000, -6.7152944, -1.6647539, 1.6569636
1: 1.9655743, 3.6435642, 1.9334784, 3.7011304, -1.3237760, 1.3394415
2: -5.4579616, -3.8526118, -5.4810061, -3.7874212, -1.2913420, 1.3005146
3: -10.1621323, -8.4082584, -10.1747808, -8.3837261, -0.9752538, 0.9723024
4: -4.7673235, -3.3324428, -4.7960691, -3.2351460, -1.3649194, 1.3682523
5: -8.3700380, -6.7941961, -8.3994455, -6.7868471, -1.1616840, 1.1729590
6: -5.9793115, -3.9414420, -5.9932337, -3.9284849, -1.6127090, 1.6155591
7: -4.2065864, -2.8140006, -4.2181315, -2.7841961, -1.3467500, 1.3338463
8: -3.7371349, -2.3019557, -3.7703476, -2.2896943, -1.1170034, 1.1223743
9: -11.0497036, -9.1602478, -11.1756716, -9.1263027, -1.4285302, 1.4151386

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6516250, upper bound: 0.6486314
time: 3.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6660866, upper bound: 0.6733527
time: 3.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8134584, -6.7280216, -1.7177491, 1.6664741
1: 1.9334784, 3.7011304, 1.9535055, 3.6440926, -1.3390920, 1.3507388
2: -5.4810061, -3.7874212, -5.4695983, -3.8512728, -1.3003223, 1.3045163
3: -10.1747808, -8.3837261, -10.1650467, -8.4066582, -1.0109477, 0.9788905
4: -4.7960691, -3.2351460, -4.7844915, -3.3316593, -1.3695536, 1.4161868
5: -8.3994455, -6.7868471, -8.3735018, -6.7897792, -1.1746044, 1.1645331
6: -5.9932337, -3.9284849, -5.9832301, -3.9411030, -1.6167383, 1.6327975
7: -4.2181315, -2.7841961, -4.2095480, -2.8125384, -1.3872893, 1.3532329
8: -3.7703476, -2.2896943, -3.7387161, -2.2968812, -1.1283584, 1.1294105
9: -11.1756716, -9.1263027, -11.0502548, -9.1371870, -1.4329598, 1.4255605

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6589527, upper bound: 0.6486312
time: 3.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6589527, upper bound: 0.6486297
time: 4.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.47 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 23.47
Output dim: 1, lower bound: -0.6516250, upper bound: 0.6486314
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.47
Output dim: 1, lower bound: -0.6660866, upper bound: 0.6733527
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 23.47
Output dim: 1, lower bound: -0.6589527, upper bound: 0.6486312
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 23.47
Output dim: 1, lower bound: -0.6589527, upper bound: 0.6486297

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8982000, -6.7152944, -1.6647501, 1.7034006
1: 1.9655747, 3.6435614, 1.9334784, 3.7011304, -1.3380239, 1.3394384
2: -5.4579601, -3.8526134, -5.4810061, -3.7874212, -1.2929757, 1.3005124
3: -10.1621304, -8.4082613, -10.1747808, -8.3837261, -0.9744561, 1.0059226
4: -4.7673187, -3.3324428, -4.7960691, -3.2351460, -1.3982115, 1.3656054
5: -8.3700352, -6.7941961, -8.3994455, -6.7868471, -1.1551354, 1.1696233
6: -5.9793115, -3.9414463, -5.9932337, -3.9284849, -1.6297078, 1.6155562
7: -4.2065811, -2.8140025, -4.2181315, -2.7841961, -1.3467500, 1.3752782
8: -3.7371349, -2.3019562, -3.7703476, -2.2896943, -1.1190138, 1.1223738
9: -11.0496979, -9.1602507, -11.1756716, -9.1263027, -1.4285250, 1.4099063

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6413217, upper bound: 0.6589517
time: 3.90 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6413216, upper bound: 0.6733510
time: 4.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.97 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.97
Output dim: 1, lower bound: -0.6413217, upper bound: 0.6589517
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.97
Output dim: 1, lower bound: -0.6413216, upper bound: 0.6733510

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8123884, -6.7421303, -8.8981972, -6.7152972, -1.7155819, 1.7033999
1: 1.9655747, 3.6435614, 1.9334812, 3.7011271, -1.3380237, 1.3607612
2: -5.4579601, -3.8526134, -5.4810052, -3.7874231, -1.2929749, 1.3044827
3: -10.1621304, -8.4082613, -10.1747808, -8.3837280, -1.0047321, 1.0059202
4: -4.7673187, -3.3324428, -4.7960649, -3.2351470, -1.3982110, 1.4027219
5: -8.3700352, -6.7941961, -8.3994436, -6.7868481, -1.1551347, 1.1637621
6: -5.9793115, -3.9414463, -5.9932308, -3.9284909, -1.6297073, 1.6376154
7: -4.2065811, -2.8140025, -4.2181268, -2.7841980, -1.3793559, 1.3752718
8: -3.7371349, -2.3019562, -3.7703452, -2.2896948, -1.1190124, 1.1235178
9: -11.0496979, -9.1602507, -11.1756668, -9.1263037, -1.4346313, 1.4099057

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6384826, upper bound: 0.6720878
time: 4.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6413175, upper bound: 0.6589468
time: 6.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.63 seconds
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.63
Output dim: 1, lower bound: -0.6384826, upper bound: 0.6720878
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.63
Output dim: 1, lower bound: -0.6413175, upper bound: 0.6589468

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.8066406, -6.7496977, -8.8948421, -6.7197723, -1.6989665, 1.6866724
1: 1.9776659, 3.6379881, 1.9405923, 3.6978846, -1.3178129, 1.3433075
2: -5.4326668, -3.8566933, -5.4661283, -3.7898006, -1.2641249, 1.2842749
3: -10.1579628, -8.4156208, -10.1722527, -8.3880692, -0.9899256, 0.9907107
4: -4.7584209, -3.3355956, -4.7908492, -3.2369876, -1.3799500, 1.3863387
5: -8.3668652, -6.8028622, -8.3976240, -6.7919664, -1.1426992, 1.1492097
6: -5.9719319, -3.9581509, -5.9888544, -3.9383397, -1.6066618, 1.6139305
7: -4.1933212, -2.8186946, -4.2102981, -2.7869782, -1.3553653, 1.3533871
8: -3.7262039, -2.3141913, -3.7637329, -2.2969055, -1.0904229, 1.0960155
9: -11.0436859, -9.1717768, -11.1721592, -9.1331120, -1.4164355, 1.3898302

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6556890, upper bound: 0.6700791
time: 4.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6557164, upper bound: 0.6720869
time: 3.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.83 seconds
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 22.83
Output dim: 1, lower bound: -0.6556890, upper bound: 0.6700791
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.83
Output dim: 1, lower bound: -0.6557164, upper bound: 0.6720869

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.8065901, -6.7497177, -8.8979034, -6.6774321, -1.7108717, 1.6890385
1: 1.9776773, 3.6379795, 1.9354520, 3.7177305, -1.3233042, 1.3492033
2: -5.4326568, -3.8566966, -5.4736204, -3.7876620, -1.2647829, 1.2910848
3: -10.1579552, -8.4156332, -10.1940012, -8.3855047, -0.9911842, 0.9978774
4: -4.7583971, -3.3356376, -4.8630037, -3.2345500, -1.3826363, 1.4056680
5: -8.3668509, -6.8028727, -8.4009018, -6.7811708, -1.1479774, 1.1536422
6: -5.9719224, -3.9581802, -6.0125656, -3.9371510, -1.6070042, 1.6221633
7: -4.1932983, -2.8187017, -4.2121696, -2.7656977, -1.3609776, 1.3548203
8: -3.7261910, -2.3141942, -3.7701607, -2.2920165, -1.0950680, 1.1029358
9: -11.0436745, -9.1717920, -11.1991014, -9.1307096, -1.4177954, 1.3965883

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6720596
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6720596
time: 3.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 22.84 seconds
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 22.84
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6720596
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.84
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6720596

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7941799, -6.7548027, -8.8978834, -6.6781683, -1.7152901, 1.6800675
1: 1.9832458, 3.6355395, 1.9355025, 3.7175455, -1.3165359, 1.3550296
2: -5.4303761, -3.8576393, -5.4733434, -3.7876749, -1.2606280, 1.2771900
3: -10.1564093, -8.4204521, -10.1938524, -8.3855228, -0.9719491, 0.9928993
4: -4.7529297, -3.3549919, -4.8619523, -3.2345645, -1.3721890, 1.3850865
5: -8.3640375, -6.8053584, -8.4008694, -6.7812591, -1.1302161, 1.1512399
6: -5.9695787, -3.9630623, -6.0124989, -3.9371676, -1.5958786, 1.6162145
7: -4.1886888, -2.8204603, -4.2121582, -2.7657356, -1.3554921, 1.3212094
8: -3.7233291, -2.3147826, -3.7700925, -2.2920413, -1.1059285, 1.1008155
9: -11.0406055, -9.1779146, -11.1989489, -9.1307373, -1.4175439, 1.3900877

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6701389
time: 3.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6720604
time: 3.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8097401, -6.7073002, -8.8979034, -6.6774321, -1.7139339, 1.6948652
1: 1.9726400, 3.6578488, 1.9354520, 3.7177305, -1.3290806, 1.3581367
2: -5.4401789, -3.8545289, -5.4736204, -3.7876620, -1.2715659, 1.2933366
3: -10.1795988, -8.4130278, -10.1940012, -8.3855047, -0.9961677, 0.9991330
4: -4.8305964, -3.3331251, -4.8630037, -3.2345500, -1.3907306, 1.4083247
5: -8.3702621, -6.7920771, -8.4009018, -6.7811708, -1.1523304, 1.1561807
6: -5.9955783, -3.9569790, -6.0125656, -3.9371510, -1.6138673, 1.6225185
7: -4.1952300, -2.7973039, -4.2121696, -2.7656977, -1.3624687, 1.3647618
8: -3.7324600, -2.3092990, -3.7701607, -2.2920165, -1.1029415, 1.1055012
9: -11.0707073, -9.1693411, -11.1991014, -9.1307096, -1.4245405, 1.3979199

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6541301
time: 6.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6701492
time: 4.01 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 25.09 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 25.09
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6701389
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 25.09
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6720604
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 25.09
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6541301
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 25.09
Output dim: 1, lower bound: -0.6537085, upper bound: 0.6701492

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.7941799, -6.7548027, -8.9067822, -6.6703186, -1.7221284, 1.6868780
1: 1.9832458, 3.6355395, 1.9240413, 3.7283096, -1.3244362, 1.3678236
2: -5.4303761, -3.8576393, -5.4920473, -3.7638447, -1.2662921, 1.2872710
3: -10.1564093, -8.4204521, -10.1991758, -8.3773050, -0.9807553, 0.9986185
4: -4.7529297, -3.3549919, -4.8704929, -3.2272153, -1.3755331, 1.3905349
5: -8.3640375, -6.8053584, -8.4105492, -6.7741084, -1.1356161, 1.1572332
6: -5.9695787, -3.9630623, -6.0337768, -3.9245915, -1.6030159, 1.6265719
7: -4.1886888, -2.8204603, -4.2232623, -2.7517035, -1.3618853, 1.3276405
8: -3.7233291, -2.3147826, -3.7904778, -2.2832327, -1.1147488, 1.1124063
9: -11.0406055, -9.1779146, -11.2125940, -9.1222420, -1.4229984, 1.3983142

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6141

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6480681, upper bound: 0.6720499
time: 3.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6536989, upper bound: 0.6720496
time: 3.88 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 28.61 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 28.61
Output dim: 1, lower bound: -0.6480681, upper bound: 0.6720499
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 28.61
Output dim: 1, lower bound: -0.6536989, upper bound: 0.6720496

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7913399, -6.8517818, -8.9066105, -6.6769342, -1.6393027, 1.5920334
1: 2.0387754, 3.6347733, 1.9280624, 3.7282715, -1.2667542, 1.3282819
2: -5.3385010, -3.8614073, -5.4859824, -3.7641051, -1.1733074, 1.2102795
3: -10.1205263, -8.4234896, -10.1964998, -8.3775158, -0.9440734, 0.9685067
4: -4.6449261, -3.3567700, -4.8628592, -3.2273149, -1.2669928, 1.3002253
5: -8.3578119, -6.8972311, -8.4099998, -6.7800298, -1.0485694, 1.0634152
6: -5.9621086, -3.9935884, -6.0330687, -3.9270928, -1.5727224, 1.5967433
7: -4.1810622, -2.8411133, -4.2225046, -2.7531402, -1.3413737, 1.3050916
8: -3.7213387, -2.3624692, -3.7903032, -2.2866077, -1.0755885, 1.0616834
9: -11.0372429, -9.2486725, -11.2123260, -9.1278906, -1.3627481, 1.3249602

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6141

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6480681, upper bound: 0.6664192
time: 3.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6480681, upper bound: 0.6720496
time: 3.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.7941818, -6.7548037, -8.9067822, -6.6703186, -1.7212620, 1.6162586
1: 1.9832458, 3.6355395, 1.9240413, 3.7283096, -1.3229198, 1.3673728
2: -5.4303761, -3.8576393, -5.4920473, -3.7638447, -1.2227497, 1.2864280
3: -10.1564083, -8.4204521, -10.1991758, -8.3773050, -0.9631093, 0.9983151
4: -4.7529292, -3.3549914, -4.8704929, -3.2272153, -1.3144541, 1.3895516
5: -8.3640385, -6.8053584, -8.4105492, -6.7741084, -1.1347530, 1.0828367
6: -5.9695787, -3.9630628, -6.0337768, -3.9245915, -1.6026812, 1.6137166
7: -4.1886883, -2.8204601, -4.2232623, -2.7517035, -1.3916099, 1.3276403
8: -3.7233295, -2.3147836, -3.7904778, -2.2832327, -1.1147484, 1.0936866
9: -11.0406055, -9.1779156, -11.2125940, -9.1222420, -1.4223804, 1.3539073

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6141

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6536991, upper bound: 0.6664199
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6536990, upper bound: 0.6720496
time: 3.80 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 28.08 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 28.08
Output dim: 1, lower bound: -0.6480681, upper bound: 0.6664192
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 28.08
Output dim: 1, lower bound: -0.6480681, upper bound: 0.6720496
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 28.08
Output dim: 1, lower bound: -0.6536991, upper bound: 0.6664199
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 28.08
Output dim: 1, lower bound: -0.6536990, upper bound: 0.6720496

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.7913399, -6.8517818, -8.9067173, -6.6712074, -1.6395068, 1.5912769
1: 2.0387754, 3.6347733, 1.9248600, 3.7283001, -1.2663226, 1.3289666
2: -5.3385010, -3.8614073, -5.4914970, -3.7639551, -1.1726036, 1.2107339
3: -10.1205263, -8.4234896, -10.1985054, -8.3773689, -0.9439313, 0.9686880
4: -4.6449261, -3.3567700, -4.8692060, -3.2272685, -1.2660487, 1.3006144
5: -8.3578119, -6.8972311, -8.4102564, -6.7744551, -1.0486672, 1.0628496
6: -5.9621086, -3.9935884, -6.0334744, -3.9254117, -1.5728025, 1.5968058
7: -4.1810622, -2.8411133, -4.2228103, -2.7519763, -1.3417335, 1.3049436
8: -3.7213387, -2.3624692, -3.7904086, -2.2838264, -1.0759053, 1.0614085
9: -11.0372429, -9.2486725, -11.2124786, -9.1238613, -1.3629713, 1.3244768

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 849

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6414973, upper bound: 0.6627500
time: 3.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6480522, upper bound: 0.6720338
time: 3.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.7941818, -6.7548037, -8.9067822, -6.6703186, -1.6514030, 1.6161506
1: 1.9832458, 3.6355395, 1.9240413, 3.7283096, -1.3224831, 1.3658736
2: -5.4303761, -3.8576393, -5.4920473, -3.7638447, -1.2223754, 1.2433541
3: -10.1564083, -8.4204521, -10.1991758, -8.3773041, -0.9629951, 0.9808593
4: -4.7529292, -3.3549914, -4.8704925, -3.2272151, -1.3141248, 1.3291290
5: -8.3640385, -6.8053584, -8.4105492, -6.7741094, -1.0611575, 1.0827734
6: -5.9695787, -3.9630628, -6.0337772, -3.9245920, -1.5904679, 1.6137166
7: -4.1886883, -2.8204601, -4.2232623, -2.7517042, -1.3916101, 1.3573709
8: -3.7233295, -2.3147836, -3.7904778, -2.2832327, -1.0960546, 1.0935075
9: -11.0406055, -9.1779156, -11.2125950, -9.1222439, -1.3784511, 1.3537605

Time for backsubstitution: 15.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 849

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6470760, upper bound: 0.6571779
time: 3.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6536831, upper bound: 0.6664030
time: 3.91 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 29.58 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 29.58
Output dim: 1, lower bound: -0.6414973, upper bound: 0.6627500
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 29.58
Output dim: 1, lower bound: -0.6480522, upper bound: 0.6720338
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 29.58
Output dim: 1, lower bound: -0.6470760, upper bound: 0.6571779
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 29.58
Output dim: 1, lower bound: -0.6536831, upper bound: 0.6664030

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.7913389, -6.8517861, -8.9066792, -6.6717472, -1.6364975, 1.5662520
1: 2.0389013, 3.6347733, 1.9252367, 3.7282925, -1.2663178, 1.3330538
2: -5.3385010, -3.8614078, -5.4914646, -3.7640171, -1.1717110, 1.2101336
3: -10.1205254, -8.4234905, -10.1981153, -8.3774080, -0.9428403, 0.9680351
4: -4.6449242, -3.3567703, -4.8687525, -3.2273023, -1.2657044, 1.3001812
5: -8.3578119, -6.8972321, -8.4100771, -6.7744808, -1.0419827, 1.0617458
6: -5.9621062, -3.9935882, -6.0333767, -3.9259167, -1.5501652, 1.5946138
7: -4.1810617, -2.8411136, -4.2225318, -2.7520170, -1.3412457, 1.3036587
8: -3.7213392, -2.3624697, -3.7903662, -2.2841163, -1.0750334, 1.0610855
9: -11.0372429, -9.2486725, -11.2124081, -9.1244793, -1.3642249, 1.3242452

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 849

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6386982, upper bound: 0.6655320
time: 3.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6386982, upper bound: 0.6720350
time: 4.32 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 29.02 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 29.02
Output dim: 1, lower bound: -0.6386982, upper bound: 0.6655320
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 29.02
Output dim: 1, lower bound: -0.6386982, upper bound: 0.6720350

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.7913389, -6.8517861, -8.9066772, -6.6717558, -1.6133060, 1.5654881
1: 2.0389013, 3.6347733, 1.9254508, 3.7282925, -1.2706864, 1.3329177
2: -5.3385010, -3.8614078, -5.4912415, -3.7640221, -1.1711903, 1.2091599
3: -10.1205254, -8.4234905, -10.1980972, -8.3774090, -0.9426777, 0.9672976
4: -4.6449242, -3.3567703, -4.8685060, -3.2273028, -1.2656658, 1.2999454
5: -8.3578119, -6.8972321, -8.4100742, -6.7746153, -1.0416377, 1.0558434
6: -5.9621062, -3.9935882, -6.0333099, -3.9259183, -1.5496578, 1.5738175
7: -4.1810617, -2.8411136, -4.2225313, -2.7521095, -1.3404117, 1.3033104
8: -3.7213392, -2.3624697, -3.7903666, -2.2841706, -1.0749147, 1.0607331
9: -11.0372429, -9.2486725, -11.2124062, -9.1247540, -1.3639894, 1.3265697

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6262442, upper bound: 0.6532080
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6386981, upper bound: 0.6464426
time: 6.47 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 32.00 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 32.00
Output dim: 1, lower bound: -0.6262442, upper bound: 0.6532080
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 13, time: 32.00
Output dim: 1, lower bound: -0.6386981, upper bound: 0.6464426
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2105.29 seconds
