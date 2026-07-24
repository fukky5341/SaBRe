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
execution time: IAR + LP analysis = 15.31 + 32.08 = 47.39 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.61 seconds, max iter: 100)

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
Binary search time: 197.32 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3355.30 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: B, layer: 1, pos: 5748
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849460
time: 3.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849392, upper bound: 0.9849404
time: 3.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.81
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849460
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.81
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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9758259
time: 3.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849404
time: 3.73 seconds

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

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9731864
time: 3.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849358, upper bound: 0.9849382
time: 3.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9758259
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9758257, upper bound: 0.9849404
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9508014, upper bound: 0.9731864
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.26
Output dim: 1, lower bound: -0.9849358, upper bound: 0.9849382

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9723887, upper bound: 0.9758280
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758142, upper bound: 0.9758268
time: 3.64 seconds

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

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758169, upper bound: 0.9815079
time: 3.71 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758146, upper bound: 0.9849347
time: 4.02 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.8939877, -6.7208433, -8.7880507, -6.7436862, -1.8689799, 1.8966155
1: 1.9399471, 3.6896782, 1.9977078, 3.5740943, -1.4546978, 1.3932219
2: -5.4793591, -3.7945230, -5.4594488, -3.8936989, -1.3904324, 1.3749216
3: -10.1730976, -8.3914833, -10.1548510, -8.4327631, -1.2192924, 1.2301878
4: -4.7839217, -3.2362547, -4.7265072, -3.3386564, -1.4452653, 1.4902525
5: -8.3859758, -6.7891879, -8.2933187, -6.8061008, -1.3414407, 1.3147395
6: -5.9841223, -3.9488535, -5.9213676, -4.0661182, -1.7143707, 1.6786277
7: -4.2052584, -2.7883868, -4.1304169, -2.8359916, -1.2991924, 1.3420300
8: -3.7683139, -2.2930851, -3.7290316, -2.3166428, -1.3106594, 1.3194211
9: -11.1536846, -9.1308994, -10.9152212, -9.1722727, -1.4503336, 1.5514765

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9697599
time: 3.63 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9731703
time: 3.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8134594, -6.7280073, -1.9709501, 1.9241571
1: 1.9334784, 3.7011304, 1.9534960, 3.6440926, -1.5318906, 1.5425267
2: -5.4810061, -3.7874212, -5.4696083, -3.8512719, -1.4385855, 1.4408555
3: -10.1747808, -8.3837261, -10.1650496, -8.4066572, -1.2927065, 1.2511629
4: -4.7960691, -3.2351460, -4.7845058, -3.3316593, -1.4644098, 1.5493598
5: -8.3994455, -6.7868471, -8.3735056, -6.7897768, -1.4055216, 1.3983358
6: -5.9932337, -3.9284849, -5.9832339, -3.9411020, -1.8470984, 1.8708143
7: -4.2181315, -2.7841961, -4.2095518, -2.8125379, -1.4055936, 1.4253557
8: -3.7703476, -2.2896943, -3.7387180, -2.2968750, -1.3353856, 1.3388789
9: -11.1756716, -9.1263027, -11.0502548, -9.1371698, -1.6779046, 1.6871378

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849358, upper bound: 0.9758241
time: 3.78 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849369, upper bound: 0.9758655
time: 3.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.43 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9723887, upper bound: 0.9758280
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9758142, upper bound: 0.9758268
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9758169, upper bound: 0.9815079
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9758146, upper bound: 0.9849347
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9697599
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9731703
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9849358, upper bound: 0.9758241
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 22.43
Output dim: 1, lower bound: -0.9849369, upper bound: 0.9758655

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472324, -8.8123894, -6.7421303, -1.8881452, 1.8767049
1: 1.9711514, 3.6410933, 1.9655743, 3.6435642, -1.4892428, 1.5010077
2: -5.4556670, -3.8535542, -5.4579616, -3.8526118, -1.4031069, 1.4004066
3: -10.1605568, -8.4130840, -10.1621323, -8.4082584, -1.2130637, 1.2254527
4: -4.7618060, -3.3518276, -4.7673235, -3.3324428, -1.4293633, 1.4154959
5: -8.3672218, -6.7966924, -8.3700380, -6.7941961, -1.3622704, 1.3798728
6: -5.9769564, -3.9463489, -5.9793115, -3.9414420, -1.8203228, 1.8242221
7: -4.2019610, -2.8157861, -4.2065864, -2.8140006, -1.3879604, 1.3908002
8: -3.7342720, -2.3025498, -3.7371349, -2.3019557, -1.3174660, 1.3039087
9: -11.0466022, -9.1663828, -11.0497036, -9.1602478, -1.6360862, 1.6287780

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9606608, upper bound: 0.9416732
time: 3.79 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9724018, upper bound: 0.9758289
time: 3.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8154755, -6.6997294, -8.8123646, -6.7421398, -1.8842163, 1.9029183
1: 1.9605546, 3.6634359, 1.9655805, 3.6435580, -1.5024811, 1.5155063
2: -5.4654770, -3.8504767, -5.4579563, -3.8526144, -1.4215279, 1.4078610
3: -10.1837940, -8.4056721, -10.1621284, -8.4082670, -1.2431488, 1.2336638
4: -4.8394833, -3.3299661, -4.7673101, -3.3324628, -1.5070205, 1.4373441
5: -8.3733959, -6.7834020, -8.3700304, -6.7942023, -1.3866844, 1.3927764
6: -6.0029597, -3.9402688, -5.9793091, -3.9414556, -1.8531442, 1.8324969
7: -4.2085056, -2.7925813, -4.2065749, -2.8140049, -1.3945007, 1.4139936
8: -3.7436275, -2.2970471, -3.7371287, -2.3019552, -1.3163741, 1.3101965
9: -11.0767317, -9.1578217, -11.0496969, -9.1602564, -1.6614122, 1.6375848

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9640708, upper bound: 0.9416724
time: 3.73 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758287, upper bound: 0.9758282
time: 3.90 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8857450, -6.7203894, -1.9030011, 1.9062858
1: 1.9655743, 3.6435642, 1.9390879, 3.6986580, -1.5187101, 1.5194236
2: -5.4579616, -3.8526118, -5.4787116, -3.7883573, -1.4209125, 1.4261792
3: -10.1621323, -8.4082584, -10.1732092, -8.3885546, -1.2426202, 1.2248385
4: -4.7673235, -3.3324428, -4.7905512, -3.2545400, -1.5127835, 1.4581084
5: -8.3700380, -6.7941961, -8.3966160, -6.7893448, -1.3874159, 1.3840003
6: -5.9793115, -3.9414420, -5.9908571, -3.9333987, -1.8367026, 1.8356293
7: -4.2065864, -2.8140006, -4.2135053, -2.7859933, -1.4205930, 1.3995047
8: -3.7371349, -2.3019557, -3.7674494, -2.2902737, -1.3198864, 1.3410652
9: -11.0497036, -9.1602478, -11.1725769, -9.1324453, -1.6617160, 1.6601186

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416595, upper bound: 0.9697657
time: 3.55 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758126, upper bound: 0.9815046
time: 3.92 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.8123646, -6.7421398, -8.9012508, -6.6729517, -1.9293022, 1.9043269
1: 1.9655805, 3.6435580, 1.9283442, 3.7209811, -1.5175719, 1.5328269
2: -5.4579563, -3.8526144, -5.4885020, -3.7852988, -1.4276831, 1.4446309
3: -10.1621284, -8.4082670, -10.1965437, -8.3811646, -1.2508757, 1.2549766
4: -4.7673101, -3.3324628, -4.8682165, -3.2327337, -1.5345764, 1.5357537
5: -8.3700304, -6.7942023, -8.4027042, -6.7760482, -1.4003282, 1.4088805
6: -5.9793091, -3.9414556, -6.0169468, -3.9272974, -1.8449993, 1.8683643
7: -4.2065749, -2.8140049, -4.2200108, -2.7628975, -1.4436774, 1.4060059
8: -3.7371287, -2.3019552, -3.7768860, -2.2847967, -1.3261237, 1.3372593
9: -11.0496969, -9.1602564, -11.2026215, -9.1239042, -1.6705441, 1.6658416

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9640523, upper bound: 0.9507875
time: 3.75 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758129, upper bound: 0.9849303
time: 3.65 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8939877, -6.7208433, -8.7755756, -6.7487993, -1.8637247, 1.9031632
1: 1.9399471, 3.6896782, 2.0031939, 3.5716200, -1.4599242, 1.3867145
2: -5.4793591, -3.7945230, -5.4571829, -3.8945978, -1.3836117, 1.3708234
3: -10.1730976, -8.3914833, -10.1532745, -8.4375668, -1.2144463, 1.2126693
4: -4.7839217, -3.2362547, -4.7214522, -3.3580630, -1.4258587, 1.4851975
5: -8.3859758, -6.7891879, -8.2904520, -6.8086624, -1.3395100, 1.2951323
6: -5.9841223, -3.9488535, -5.9188776, -4.0710287, -1.7085280, 1.6689663
7: -4.2052584, -2.7883868, -4.1257935, -2.8377655, -1.2772655, 1.3374066
8: -3.7683139, -2.2930851, -3.7261791, -2.3172345, -1.3090973, 1.3314747
9: -11.1536846, -9.1308994, -10.9121227, -9.1784077, -1.4440055, 1.5524590

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9473788
time: 3.61 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9697599
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.8939629, -6.7208533, -8.7911415, -6.7011423, -1.8744683, 1.8984025
1: 1.9399538, 3.6896739, 1.9928570, 3.5940027, -1.4746668, 1.3998947
2: -5.4793539, -3.7945247, -5.4669089, -3.8915477, -1.3909979, 1.3813508
3: -10.1730957, -8.3914890, -10.1765118, -8.4302416, -1.2225705, 1.2350490
4: -4.7839098, -3.2362757, -4.7993445, -3.3361793, -1.4477305, 1.5311284
5: -8.3859692, -6.7891941, -8.2967396, -6.7952881, -1.3437982, 1.3198308
6: -5.9841170, -3.9488692, -5.9451299, -4.0649395, -1.7168050, 1.6848872
7: -4.2052460, -2.7883916, -4.1323686, -2.8145552, -1.3042860, 1.3439770
8: -3.7683077, -2.2930870, -3.7353559, -2.3116970, -1.3127961, 1.3302397
9: -11.1536789, -9.1309071, -10.9423370, -9.1699429, -1.4526522, 1.5699251

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9507843
time: 3.68 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9731703
time: 3.64 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8123884, -6.7421303, -1.9566007, 1.9081829
1: 1.9334784, 3.7011304, 1.9655747, 3.6435614, -1.5259299, 1.5298083
2: -5.4810061, -3.7874212, -5.4579601, -3.8526134, -1.4302995, 1.4293139
3: -10.1747808, -8.3837261, -10.1621304, -8.4082613, -1.2819953, 1.2467270
4: -4.7960691, -3.2351460, -4.7673187, -3.3324428, -1.4636264, 1.5321727
5: -8.3994455, -6.7868471, -8.3700352, -6.7941961, -1.4005394, 1.3840576
6: -5.9932337, -3.9284849, -5.9793115, -3.9414463, -1.8453922, 1.8677220
7: -4.2181315, -2.7841961, -4.2065811, -2.8140025, -1.4041290, 1.4223850
8: -3.7703476, -2.2896943, -3.7371349, -2.3019562, -1.3293996, 1.3238077
9: -11.1756716, -9.1263027, -11.0496979, -9.1602507, -1.6548495, 1.6680324

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9758140
time: 3.85 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849244, upper bound: 0.9758130
time: 3.74 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8981972, -6.7152972, -1.9836769, 1.9287872
1: 1.9334784, 3.7011304, 1.9334812, 3.7011271, -1.5424299, 1.5604732
2: -5.4810061, -3.7874212, -5.4810052, -3.7874231, -1.4499912, 1.4524298
3: -10.1747808, -8.3837261, -10.1747808, -8.3837280, -1.2964687, 1.2599164
4: -4.7960691, -3.2351460, -4.7960649, -3.2351470, -1.5609221, 1.5609188
5: -8.3994455, -6.7868471, -8.3994436, -6.7868481, -1.4102113, 1.4060577
6: -5.9932337, -3.9284849, -5.9932308, -3.9284909, -1.8563237, 1.8819391
7: -4.2181315, -2.7841961, -4.2181268, -2.7841980, -1.4339335, 1.4339306
8: -3.7703476, -2.2896943, -3.7703452, -2.2896948, -1.3433187, 1.3456564
9: -11.1756716, -9.1263027, -11.1756668, -9.1263037, -1.6887317, 1.6923927

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849267, upper bound: 0.9724271
time: 3.76 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849258, upper bound: 0.9758541
time: 3.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.41 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9606608, upper bound: 0.9416732
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9724018, upper bound: 0.9758289
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9640708, upper bound: 0.9416724
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9758287, upper bound: 0.9758282
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9416595, upper bound: 0.9697657
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9758126, upper bound: 0.9815046
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9640523, upper bound: 0.9507875
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9758129, upper bound: 0.9849303
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9473788
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9697599
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9507843
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9731703
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9758140
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9849244, upper bound: 0.9758130
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9849267, upper bound: 0.9724271
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 22.41
Output dim: 1, lower bound: -0.9849258, upper bound: 0.9758541

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.7744923, -6.7630730, -8.8080835, -6.7458239, -1.8614502, 1.8350277
1: 2.0152016, 3.5710583, 1.9720440, 3.6320763, -1.3720155, 1.4234874
2: -5.4455094, -3.8960195, -5.4562645, -3.8598168, -1.3487983, 1.3521988
3: -10.1503220, -8.4390316, -10.1604261, -8.4160604, -1.1926653, 1.1977267
4: -4.7044754, -3.3588679, -4.7553225, -3.3335886, -1.3708868, 1.3964546
5: -8.2867794, -6.8130946, -8.3562202, -6.7966118, -1.2736144, 1.3220329
6: -5.9148703, -4.0713606, -5.9702334, -3.9617968, -1.6580987, 1.6920519
7: -4.1227598, -2.8388700, -4.1936898, -2.8174853, -1.3052745, 1.2624111
8: -3.7246737, -2.3223124, -3.7351398, -2.3053474, -1.3029146, 1.2786735
9: -10.9115448, -9.2015972, -11.0276880, -9.1652317, -1.5004992, 1.4152321

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9382713, upper bound: 0.9416735
time: 3.62 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9382713, upper bound: 0.9416731
time: 3.83 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472343, -8.8123894, -6.7421303, -1.8881423, 1.9357390
1: 1.9711537, 3.6410923, 1.9655743, 3.6435642, -1.5140061, 1.5010043
2: -5.4556665, -3.8535562, -5.4579616, -3.8526118, -1.4077182, 1.4004040
3: -10.1605568, -8.4130869, -10.1621323, -8.4082584, -1.2122591, 1.2652460
4: -4.7618022, -3.3518291, -4.7673235, -3.3324428, -1.4293594, 1.4154944
5: -8.3672190, -6.7966933, -8.3700380, -6.7941961, -1.3569686, 1.3798726
6: -5.9769545, -3.9463549, -5.9793115, -3.9414420, -1.8459392, 1.8242195
7: -4.2019572, -2.8157864, -4.2065864, -2.8140006, -1.3879566, 1.3908000
8: -3.7342701, -2.3025527, -3.7371349, -2.3019557, -1.3198023, 1.3039072
9: -11.0465975, -9.1663837, -11.0497036, -9.1602478, -1.6360810, 1.6358614

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9724018, upper bound: 0.9724037
time: 3.74 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9724018, upper bound: 0.9758289
time: 3.79 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.7900629, -6.7154074, -8.8080568, -6.7458353, -1.8567243, 1.8551576
1: 2.0048685, 3.5934381, 1.9720507, 3.6320724, -1.3852108, 1.4383206
2: -5.4552307, -3.8929613, -5.4562583, -3.8598185, -1.3643351, 1.3596873
3: -10.1735210, -8.4316940, -10.1604223, -8.4160681, -1.2228885, 1.2058599
4: -4.7823601, -3.3369749, -4.7553110, -3.3336091, -1.4487510, 1.4183362
5: -8.2931137, -6.7997208, -8.3562126, -6.7966170, -1.2983248, 1.3263198
6: -5.9411144, -4.0652766, -5.9702282, -3.9618108, -1.6740189, 1.7003262
7: -4.1293292, -2.8156500, -4.1936793, -2.8174901, -1.3118391, 1.2893293
8: -3.7338781, -2.3167763, -3.7351341, -2.3053474, -1.3018055, 1.2850602
9: -10.9417534, -9.1931400, -11.0276823, -9.1652412, -1.5253372, 1.4232599

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416733, upper bound: 0.9416727
time: 3.85 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416720, upper bound: 0.9416720
time: 4.16 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.8154745, -6.6997290, -8.8123646, -6.7421398, -1.8842130, 1.9575655
1: 1.9605546, 3.6634345, 1.9655805, 3.6435580, -1.5272436, 1.5155060
2: -5.4654770, -3.8504791, -5.4579563, -3.8526144, -1.4261398, 1.4078588
3: -10.1837902, -8.4056749, -10.1621284, -8.4082670, -1.2423484, 1.2734050
4: -4.8394790, -3.3299670, -4.7673101, -3.3324628, -1.5070162, 1.4373431
5: -8.3733931, -6.7834034, -8.3700304, -6.7942023, -1.3813822, 1.3909484
6: -6.0029569, -3.9402728, -5.9793091, -3.9414556, -1.8685670, 1.8324931
7: -4.2085009, -2.7925828, -4.2065749, -2.8140049, -1.3944960, 1.4139922
8: -3.7436285, -2.2970476, -3.7371287, -2.3019552, -1.3187099, 1.3101954
9: -11.0767269, -9.1578236, -11.0496969, -9.1602564, -1.6614118, 1.6446683

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758287, upper bound: 0.9724008
time: 3.68 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9758290, upper bound: 0.9725029
time: 3.97 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8080835, -6.7458239, -8.8606462, -6.7392435, -1.8588271, 1.8790097
1: 1.9720440, 3.6320763, 1.9856119, 3.6284451, -1.4411006, 1.4003317
2: -5.4562645, -3.8598168, -5.4671717, -3.8306851, -1.3706145, 1.3713481
3: -10.1604261, -8.4160604, -10.1627960, -8.4186106, -1.2083297, 1.2038801
4: -4.7553225, -3.3335886, -4.7299571, -3.2615037, -1.4938188, 1.3963685
5: -8.3562202, -6.7966118, -8.3170090, -6.8068700, -1.3287380, 1.2946090
6: -5.9702334, -3.9617968, -5.9250436, -4.0585179, -1.7042842, 1.6682305
7: -4.1936898, -2.8174853, -4.1342411, -2.8145847, -1.2700043, 1.3167558
8: -3.7351398, -2.3053474, -3.7560740, -2.3104076, -1.2943003, 1.3219638
9: -11.0276880, -9.1652317, -11.0373001, -9.1694813, -1.4508848, 1.5235748

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416595, upper bound: 0.9473821
time: 3.62 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416595, upper bound: 0.9697656
time: 3.66 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -8.8123894, -6.7421303, -8.8857412, -6.7203903, -1.9620266, 1.9062855
1: 1.9655743, 3.6435642, 1.9390893, 3.6986542, -1.5187094, 1.5441864
2: -5.4579616, -3.8526118, -5.4787121, -3.7883589, -1.4209113, 1.4307904
3: -10.1621323, -8.4082584, -10.1732092, -8.3885574, -1.2783744, 1.2241329
4: -4.7673235, -3.3324428, -4.7905445, -3.2545404, -1.5127831, 1.4581017
5: -8.3700380, -6.7941961, -8.3966150, -6.7893462, -1.3874154, 1.3786981
6: -5.9793115, -3.9414420, -5.9908557, -3.9334023, -1.8366995, 1.8612453
7: -4.2065864, -2.8140006, -4.2135000, -2.7859943, -1.4205921, 1.3994994
8: -3.7371349, -2.3019557, -3.7674475, -2.2902737, -1.3198850, 1.3425376
9: -11.0497036, -9.1602478, -11.1725712, -9.1324444, -1.6687999, 1.6601183

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9723874, upper bound: 0.9815047
time: 3.82 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9723874, upper bound: 0.9815058
time: 3.71 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.7869415, -6.7579679, -8.8970461, -6.6784811, -1.9020572, 1.8563139
1: 2.0097218, 3.5735250, 1.9348612, 3.7095351, -1.3860745, 1.4555240
2: -5.4477701, -3.8951211, -5.4868460, -3.7923934, -1.3640492, 1.3964801
3: -10.1518936, -8.4342365, -10.1948538, -8.3889236, -1.2289343, 1.2237417
4: -4.7095180, -3.3394799, -4.8561168, -3.2338333, -1.4756846, 1.5166368
5: -8.2896433, -6.8105388, -8.3892536, -6.7783899, -1.3116500, 1.3411412
6: -5.9173541, -4.0664659, -6.0078516, -3.9476650, -1.6766303, 1.7361858
7: -4.1273680, -2.8371038, -4.2071409, -2.7670946, -1.3602734, 1.2976496
8: -3.7275219, -2.3217220, -3.7748098, -2.2881823, -1.3108290, 1.3124895
9: -10.9146318, -9.1954746, -11.1806469, -9.1285152, -1.5347705, 1.4328275

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416586, upper bound: 0.9507875
time: 3.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9416586, upper bound: 0.9507887
time: 3.93 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8123627, -6.7421417, -8.9012508, -6.6729517, -1.9293017, 1.9589839
1: 1.9655819, 3.6435566, 1.9283442, 3.7209811, -1.5352678, 1.5328236
2: -5.4579558, -3.8526161, -5.4885020, -3.7852988, -1.4299591, 1.4446290
3: -10.1621275, -8.4082680, -10.1965437, -8.3811646, -1.2500784, 1.2922258
4: -4.7673068, -3.3324633, -4.8682165, -3.2327337, -1.5345731, 1.5357533
5: -8.3700285, -6.7942009, -8.4027042, -6.7760482, -1.3950269, 1.4055450
6: -5.9793067, -3.9414616, -6.0169468, -3.9272974, -1.8701797, 1.8683641
7: -4.2065701, -2.8140061, -4.2200108, -2.7628975, -1.4436727, 1.4060047
8: -3.7371273, -2.3019576, -3.7768860, -2.2847967, -1.3284595, 1.3372586
9: -11.0496922, -9.1602592, -11.2026215, -9.1239042, -1.6705389, 1.6615977

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9723857, upper bound: 0.9849305
time: 3.81 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9723857, upper bound: 0.9816104
time: 4.08 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.8731165, -6.7341380, -8.7755756, -6.7487993, -1.8502603, 1.8689275
1: 1.9800858, 3.6309133, 2.0031939, 3.5716200, -1.3845013, 1.3624895
2: -5.4694366, -3.8297918, -5.4571829, -3.8945978, -1.3479255, 1.3482177
3: -10.1643639, -8.4138031, -10.1532745, -8.4375668, -1.2048078, 1.1916353
4: -4.7350035, -3.2420909, -4.7214522, -3.3580630, -1.3769405, 1.4793613
5: -8.3198881, -6.8043079, -8.2904520, -6.8086624, -1.2924728, 1.2719910
6: -5.9275594, -4.0536022, -5.9188776, -4.0710287, -1.6313167, 1.6191785
7: -4.1388626, -2.8128054, -4.1257935, -2.8377655, -1.2339177, 1.2560329
8: -3.7589579, -2.3098330, -3.7261791, -2.3172345, -1.3000734, 1.3128703
9: -11.0403852, -9.1633463, -10.9121227, -9.1784077, -1.3955362, 1.4161186

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9382563
time: 3.77 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507854, upper bound: 0.9382563
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8980713, -6.7177715, -8.7755756, -6.7487993, -1.8654037, 1.9019022
1: 1.9348330, 3.7009501, 2.0031939, 3.5716200, -1.4632401, 1.3869464
2: -5.4801965, -3.7876968, -5.4571829, -3.8945978, -1.3838775, 1.3735688
3: -10.1746130, -8.3870163, -10.1532745, -8.4375668, -1.2160504, 1.2145625
4: -4.7942004, -3.2352190, -4.7214522, -3.3580630, -1.4361374, 1.4862332
5: -8.3987799, -6.7876205, -8.2904520, -6.8086624, -1.3450933, 1.2964540
6: -5.9912491, -3.9284964, -5.9188776, -4.0710287, -1.7120576, 1.6693811
7: -4.2180510, -2.7863302, -4.1257935, -2.8377655, -1.2783232, 1.3394632
8: -3.7692409, -2.2899036, -3.7261791, -2.3172345, -1.3093281, 1.3349837
9: -11.1754999, -9.1283960, -10.9121227, -9.1784077, -1.4452546, 1.5505103

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9473808, upper bound: 0.9697601
time: 3.56 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9473808, upper bound: 0.9697598
time: 3.80 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -8.8730898, -6.7341475, -8.7911415, -6.7011423, -1.8610015, 1.8663688
1: 1.9800911, 3.6309080, 1.9928570, 3.5940027, -1.3959696, 1.3756697
2: -5.4694319, -3.8297944, -5.4669089, -3.8915477, -1.3554044, 1.3587446
3: -10.1643600, -8.4138145, -10.1765118, -8.4302416, -1.2129322, 1.2138733
4: -4.7349939, -3.2421112, -4.7993445, -3.3361793, -1.3988147, 1.5194972
5: -8.3198795, -6.8043113, -8.2967396, -6.7952881, -1.2967606, 1.2965329
6: -5.9275537, -4.0536170, -5.9451299, -4.0649395, -1.6396036, 1.6433904
7: -4.1388512, -2.8128097, -4.1323686, -2.8145552, -1.2737474, 1.2640306
8: -3.7589512, -2.3098335, -3.7353559, -2.3116970, -1.3037806, 1.3116221
9: -11.0403786, -9.1633549, -10.9423370, -9.1699429, -1.4041839, 1.4262384

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9416583
time: 3.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507858, upper bound: 0.9416586
time: 3.71 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8980455, -6.7177806, -8.7911415, -6.7011423, -1.8761468, 1.8971860
1: 1.9348402, 3.7009449, 1.9928570, 3.5940027, -1.4728301, 1.4001272
2: -5.4801912, -3.7876987, -5.4669089, -3.8915477, -1.3912735, 1.3840957
3: -10.1746092, -8.3870239, -10.1765118, -8.4302416, -1.2241359, 1.2369421
4: -4.7941875, -3.2352395, -4.7993445, -3.3361793, -1.4580083, 1.5273175
5: -8.3987722, -6.7876253, -8.2967396, -6.7952881, -1.3493810, 1.3211788
6: -5.9912438, -3.9285104, -5.9451299, -4.0649395, -1.7203345, 1.6853025
7: -4.2180395, -2.7863343, -4.1323686, -2.8145552, -1.3053617, 1.3460343
8: -3.7692347, -2.2899060, -3.7353559, -2.3116970, -1.3130269, 1.3328931
9: -11.1754942, -9.1284027, -10.9423370, -9.1699429, -1.4539018, 1.5603839

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9640519
time: 3.87 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9507846, upper bound: 0.9640519
time: 3.70 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.8857450, -6.7203894, -8.8123884, -6.7421303, -1.9609432, 1.9029977
1: 1.9390879, 3.6986580, 1.9655747, 3.6435614, -1.5194204, 1.5364060
2: -5.4787116, -3.7883573, -5.4579601, -3.8526134, -1.4261770, 1.4231875
3: -10.1732092, -8.3885546, -10.1621304, -8.4082613, -1.2647450, 1.2418227
4: -4.7905512, -3.2545400, -4.7673187, -3.3324428, -1.4581084, 1.5127788
5: -8.3966160, -6.7893448, -8.3700352, -6.7941961, -1.3806646, 1.3821137
6: -5.9908571, -3.9333987, -5.9793115, -3.9414463, -1.8356256, 1.8618622
7: -4.2135053, -2.7859933, -4.2065811, -2.8140025, -1.3995028, 1.4205878
8: -3.7674494, -2.2902737, -3.7371349, -2.3019562, -1.3410654, 1.3222215
9: -11.1725769, -9.1324453, -11.0496979, -9.1602507, -1.6558747, 1.6617112

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9815033, upper bound: 0.9723858
time: 3.65 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9815033, upper bound: 0.9758131
time: 3.81 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9012508, -6.6729517, -8.8123627, -6.7421417, -1.9589839, 1.9293017
1: 1.9283442, 3.7209811, 1.9655819, 3.6435566, -1.5328236, 1.5352678
2: -5.4885020, -3.7852988, -5.4579558, -3.8526161, -1.4446292, 1.4299588
3: -10.1965437, -8.3811646, -10.1621275, -8.4082680, -1.2922261, 1.2500784
4: -4.8682165, -3.2327337, -4.7673068, -3.3324633, -1.5357533, 1.5345731
5: -8.4027042, -6.7760482, -8.3700285, -6.7942009, -1.4055450, 1.3950268
6: -6.0169468, -3.9272974, -5.9793067, -3.9414616, -1.8683643, 1.8701797
7: -4.2200108, -2.7628975, -4.2065701, -2.8140061, -1.4060047, 1.4436727
8: -3.7768860, -2.2847967, -3.7371273, -2.3019576, -1.3372588, 1.3284594
9: -11.2026215, -9.1239042, -11.0496922, -9.1602592, -1.6615975, 1.6705389

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849306, upper bound: 0.9723858
time: 3.82 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849306, upper bound: 0.9758130
time: 3.85 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -8.8982000, -6.7152944, -8.8857412, -6.7203903, -1.9784517, 1.9332085
1: 1.9334784, 3.7011304, 1.9390893, 3.6986542, -1.5490277, 1.5538161
2: -5.4810061, -3.7874212, -5.4787121, -3.7883589, -1.4439716, 1.4482932
3: -10.1747808, -8.3837261, -10.1732092, -8.3885574, -1.2915647, 1.2424320
4: -4.7960691, -3.2351460, -4.7905445, -3.2545404, -1.5415287, 1.5553985
5: -8.3994455, -6.7868471, -8.3966150, -6.7893462, -1.4083023, 1.3864874
6: -5.9932337, -3.9284849, -5.9908557, -3.9334023, -1.8504815, 1.8721972
7: -4.2181315, -2.7841961, -4.2135000, -2.7859943, -1.4321373, 1.4293039
8: -3.7703476, -2.2896943, -3.7674475, -2.2902737, -1.3416944, 1.3578324
9: -11.1756716, -9.1263027, -11.1725712, -9.1324444, -1.6823947, 1.6934183

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9724272
time: 3.99 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9724274
time: 4.10 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -8.8981724, -6.7153053, -8.9012489, -6.6729527, -1.9893589, 1.9311697
1: 1.9334846, 3.7011256, 1.9283471, 3.7209787, -1.5478892, 1.5673227
2: -5.4810009, -3.7874227, -5.4885015, -3.7853010, -1.4506366, 1.4590373
3: -10.1747761, -8.3837337, -10.1965418, -8.3811674, -1.2997813, 1.2650559
4: -4.7960563, -3.2351665, -4.8682117, -3.2327337, -1.5633225, 1.6330452
5: -8.3994398, -6.7868524, -8.4027014, -6.7760501, -1.4127343, 1.4110647
6: -5.9932289, -3.9284992, -6.0169439, -3.9273019, -1.8587782, 1.8903389
7: -4.2181215, -2.7842007, -4.2200041, -2.7628992, -1.4552224, 1.4358034
8: -3.7703409, -2.2896957, -3.7768850, -2.2847977, -1.3479226, 1.3561265
9: -11.1756649, -9.1263094, -11.2026138, -9.1239052, -1.6912670, 1.6991405

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9758543
time: 3.73 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9758540
time: 4.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.78 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9382713, upper bound: 0.9416735
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9382713, upper bound: 0.9416731
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9724018, upper bound: 0.9724037
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9724018, upper bound: 0.9758289
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9416733, upper bound: 0.9416727
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9416720, upper bound: 0.9416720
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9758287, upper bound: 0.9724008
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9758290, upper bound: 0.9725029
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9416595, upper bound: 0.9473821
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9416595, upper bound: 0.9697656
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9723874, upper bound: 0.9815047
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9723874, upper bound: 0.9815058
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9416586, upper bound: 0.9507875
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9416586, upper bound: 0.9507887
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9723857, upper bound: 0.9849305
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9723857, upper bound: 0.9816104
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9507844, upper bound: 0.9382563
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9507854, upper bound: 0.9382563
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9473808, upper bound: 0.9697601
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9473808, upper bound: 0.9697598
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9416583
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9507858, upper bound: 0.9416586
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9507835, upper bound: 0.9640519
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9507846, upper bound: 0.9640519
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9815033, upper bound: 0.9723858
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9815033, upper bound: 0.9758131
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9849306, upper bound: 0.9723858
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9849306, upper bound: 0.9758130
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9724272
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9724274
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9758543
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.78
Output dim: 1, lower bound: -0.9814974, upper bound: 0.9758540

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7744923, -6.7630730, -8.7869673, -6.7579565, -1.8279376, 1.8159883
1: 2.0152016, 3.5710583, 2.0097160, 3.5735288, -1.3369102, 1.3485801
2: -5.4455094, -3.8960195, -5.4477758, -3.8951197, -1.3185387, 1.3158853
3: -10.1503220, -8.4390316, -10.1518946, -8.4342260, -1.1763363, 1.1887312
4: -4.7044754, -3.3588679, -4.7095294, -3.3394604, -1.3650150, 1.3506615
5: -8.2867794, -6.8130946, -8.2896528, -6.8105326, -1.2482979, 1.2657754
6: -5.9148703, -4.0713606, -5.9173589, -4.0664520, -1.6017013, 1.6055480
7: -4.1227598, -2.8388700, -4.1273794, -2.8371000, -1.2277033, 1.2143866
8: -3.7246737, -2.3223124, -3.7275267, -2.3217211, -1.2844857, 1.2710053
9: -10.9115448, -9.2015972, -10.9146385, -9.1954670, -1.3560420, 1.3487315

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9382713, upper bound: 0.9382732
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9382713, upper bound: 0.9416735
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7744923, -6.7630730, -8.8123770, -6.7423153, -1.8597767, 1.8408558
1: 2.0152016, 3.5710583, 1.9656744, 3.6435490, -1.3725216, 1.4285530
2: -5.4455094, -3.8960195, -5.4579010, -3.8526344, -1.3575368, 1.3528013
3: -10.1503220, -8.4390316, -10.1621189, -8.4084911, -1.2001824, 1.1997132
4: -4.7044754, -3.3588679, -4.7671833, -3.3324485, -1.3720269, 1.4083154
5: -8.2867794, -6.8130946, -8.3699846, -6.7942533, -1.2753847, 1.3284223
6: -5.9148703, -4.0713606, -5.9791684, -3.9414470, -1.6585681, 1.6980197
7: -4.1227598, -2.8388700, -4.2065754, -2.8141119, -1.3086479, 1.2635667
8: -3.7246737, -2.3223124, -3.7370596, -2.3019710, -1.3066800, 1.2807167
9: -10.9115448, -9.2015972, -11.0496855, -9.1604042, -1.5018873, 1.4167514

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9382712, upper bound: 0.9382726
time: 3.97 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9382712, upper bound: 0.9416729
time: 4.49 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472343, -8.7999287, -6.7472324, -1.8819594, 1.9409971
1: 1.9711537, 3.6410923, 1.9711514, 3.6410933, -1.5192935, 1.4945270
2: -5.4556665, -3.8535562, -5.4556670, -3.8535542, -1.4001575, 1.3955445
3: -10.1605568, -8.4130869, -10.1605568, -8.4130840, -1.2073557, 1.2479749
4: -4.7618022, -3.3518291, -4.7618060, -3.3518276, -1.4099746, 1.4099770
5: -8.3672190, -6.7966933, -8.3672218, -6.7966924, -1.3552482, 1.3605492
6: -5.9769545, -3.9463549, -5.9769564, -3.9463489, -1.8401003, 1.8144796
7: -4.2019572, -2.8157864, -4.2019610, -2.8157861, -1.3861711, 1.3861747
8: -3.7342701, -2.3025527, -3.7342720, -2.3025498, -1.3186302, 1.3162929
9: -11.0465975, -9.1663837, -11.0466022, -9.1663828, -1.6297369, 1.6368248

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9670050, upper bound: 0.9714811
time: 3.69 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9723958, upper bound: 0.9723981
time: 3.71 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472343, -8.8154755, -6.6997294, -1.9047842, 1.9380727
1: 1.9711537, 3.6410923, 1.9605546, 3.6634359, -1.5266452, 1.5072476
2: -5.4556665, -3.8535562, -5.4654770, -3.8504767, -1.4083619, 1.4054213
3: -10.1605568, -8.4130869, -10.1837940, -8.4056721, -1.2144954, 1.2754003
4: -4.7618022, -3.3518291, -4.8394833, -3.3299661, -1.4318361, 1.4876542
5: -8.3672190, -6.7966933, -8.3733959, -6.7834020, -1.3680661, 1.3844664
6: -5.9769545, -3.9463549, -6.0029597, -3.9402688, -1.8472590, 1.8472891
7: -4.2019572, -2.8157864, -4.2085056, -2.7925813, -1.4093759, 1.3927193
8: -3.7342701, -2.3025527, -3.7436275, -2.2970471, -1.3250115, 1.3103175
9: -11.0465975, -9.1663837, -11.0767317, -9.1578217, -1.6379445, 1.6508096

Time for backsubstitution: 14.67 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.509117841720581
rel_dist={1: [-0.9849775715086642, 0.9849766529340247]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554961, upper bound: 0.7464840
time: 3.72 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554972, upper bound: 0.7554952
time: 3.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.59 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 7.59
Output dim: 1, lower bound: -0.7554961, upper bound: 0.7464840
IS_B2, status: Status.UNKNOWN, split count: 1, time: 7.59
Output dim: 1, lower bound: -0.7554972, upper bound: 0.7554952

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.8134632, -6.7279902, -8.8123894, -6.7421303, -1.7005167, 1.7136462
1: 1.9534850, 3.6440954, 1.9655743, 3.6435642, -1.3685148, 1.3565863
2: -5.4696188, -3.8512685, -5.4579616, -3.8526118, -1.3213863, 1.3111424
3: -10.1650553, -8.4066534, -10.1621323, -8.4082584, -1.0324852, 1.0299275
4: -4.7845268, -3.3316565, -4.7673235, -3.3324428, -1.4060545, 1.3887913
5: -8.3735094, -6.7897739, -8.3700380, -6.7941961, -1.2166901, 1.2160375
6: -5.9832397, -3.9410968, -5.9793115, -3.9414420, -1.6607938, 1.6582832
7: -4.2095613, -2.8125353, -4.2065864, -2.8140006, -1.3605847, 1.3572133
8: -3.7387199, -2.2968702, -3.7371349, -2.3019557, -1.1542066, 1.1581008
9: -11.0502615, -9.1371479, -11.0497036, -9.1602478, -1.4562726, 1.4785833

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7464846
time: 3.77 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7464845
time: 3.79 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.8134613, -6.7280145, -8.8982000, -6.7152944, -1.7308984, 1.7325675
1: 1.9534998, 3.6440954, 1.9334784, 3.7011304, -1.3835762, 1.3872950
2: -5.4696021, -3.8512707, -5.4810061, -3.7874212, -1.3368156, 1.3348904
3: -10.1650486, -8.4066563, -10.1747808, -8.3837261, -1.0477846, 1.0489734
4: -4.7845011, -3.3316588, -4.7960691, -3.2351460, -1.4320772, 1.4249089
5: -8.3735046, -6.7897778, -8.3994455, -6.7868471, -1.2292194, 1.2356712
6: -5.9832325, -3.9410970, -5.9932337, -3.9284849, -1.6732726, 1.6743313
7: -4.2095566, -2.8125374, -4.2181315, -2.7841961, -1.3868468, 1.3900738
8: -3.7387185, -2.2968788, -3.7703476, -2.2896943, -1.1796850, 1.1801157
9: -11.0502615, -9.1371822, -11.1756716, -9.1263027, -1.4909601, 1.4992070

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554788, upper bound: 0.7528342
time: 3.69 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554947, upper bound: 0.7554924
time: 3.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.23 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7464846
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 1, lower bound: -0.7464855, upper bound: 0.7464845
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 1, lower bound: -0.7554788, upper bound: 0.7528342
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 1, lower bound: -0.7554947, upper bound: 0.7554924

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438317, upper bound: 0.7464671
time: 3.51 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464874, upper bound: 0.7464814
time: 3.79 seconds

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438317, upper bound: 0.7464671
time: 3.86 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464889, upper bound: 0.7464820
time: 4.02 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -8.8115549, -6.7287598, -8.8857450, -6.7203894, -1.7237468, 1.7378426
1: 1.9543409, 3.6437254, 1.9390879, 3.6986580, -1.3889837, 1.3801557
2: -5.4692640, -3.8514154, -5.4787116, -3.7883573, -1.3256967, 1.3303509
3: -10.1648140, -8.4073811, -10.1732092, -8.3885546, -1.0425005, 1.0298302
4: -4.7835989, -3.3345814, -4.7905512, -3.2545400, -1.4123702, 1.4186602
5: -8.3730764, -6.7901425, -8.3966160, -6.7893448, -1.2267623, 1.2170911
6: -5.9828787, -3.9418395, -5.9908571, -3.9333987, -1.6670964, 1.6630685
7: -4.2088566, -2.8128064, -4.2135053, -2.7859933, -1.3524566, 1.3841777
8: -3.7382755, -2.2969670, -3.7674494, -2.2902737, -1.1774319, 1.1904588
9: -11.0498028, -9.1381092, -11.1725769, -9.1324453, -1.4842501, 1.4992375

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528363, upper bound: 0.7528344
time: 3.76 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528363, upper bound: 0.7528340
time: 5.17 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -8.8134193, -6.7280321, -8.9012508, -6.6729517, -1.7478933, 1.7349455
1: 1.9535103, 3.6440878, 1.9283442, 3.7209811, -1.3890319, 1.3934362
2: -5.4695940, -3.8512735, -5.4885020, -3.7852988, -1.3374569, 1.3482494
3: -10.1650438, -8.4066696, -10.1965437, -8.3811646, -1.0495961, 1.0541034
4: -4.7844815, -3.3316927, -4.8682165, -3.2327337, -1.4348440, 1.4461772
5: -8.3734922, -6.7897868, -8.4027042, -6.7760482, -1.2354620, 1.2402499
6: -5.9832253, -3.9411221, -6.0169468, -3.9272974, -1.6741467, 1.6932747
7: -4.2095356, -2.8125443, -4.2200108, -2.7628975, -1.3924689, 1.3917639
8: -3.7387066, -2.2968798, -3.7768860, -2.2847967, -1.1842847, 1.1872747
9: -11.0502529, -9.1371956, -11.2026215, -9.1239042, -1.4925952, 1.5059540

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464830, upper bound: 0.7554934
time: 3.81 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464814, upper bound: 0.7465578
time: 3.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.37 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7438317, upper bound: 0.7464671
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7464874, upper bound: 0.7464814
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7438317, upper bound: 0.7464671
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7464889, upper bound: 0.7464820
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7528363, upper bound: 0.7528344
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7528363, upper bound: 0.7528340
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7464830, upper bound: 0.7554934
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 1, lower bound: -0.7464814, upper bound: 0.7465578

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472324, -8.8104820, -6.7428751, -1.7064142, 1.6921840
1: 1.9711514, 3.6410933, 1.9664164, 3.6431956, -1.3487449, 1.3603181
2: -5.4556670, -3.8535542, -5.4576225, -3.8527575, -1.3053496, 1.2985013
3: -10.1605568, -8.4130840, -10.1618958, -8.4089861, -1.0091981, 1.0226892
4: -4.7618060, -3.3518276, -4.7664204, -3.3353643, -1.3817427, 1.3683012
5: -8.3672218, -6.7966924, -8.3696098, -6.7945609, -1.1928067, 1.2086036
6: -5.9769564, -3.9463489, -5.9789581, -3.9421825, -1.6464219, 1.6515127
7: -4.2019610, -2.8157861, -4.2058878, -2.8142688, -1.3481572, 1.3234880
8: -3.7342720, -2.3025498, -3.7366943, -2.3020444, -1.1627795, 1.1499289
9: -11.0466022, -9.1663828, -11.0492449, -9.1611757, -1.4556177, 1.4487593

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438337, upper bound: 0.7438328
time: 3.67 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438337, upper bound: 0.7464753
time: 3.63 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.8154755, -6.6997294, -8.8123455, -6.7421470, -1.7016339, 1.7187512
1: 1.9605546, 3.6634359, 1.9655838, 3.6435561, -1.3618603, 1.3742497
2: -5.4654770, -3.8504767, -5.4579525, -3.8526151, -1.3232174, 1.3105145
3: -10.1837940, -8.4056721, -10.1621256, -8.4082718, -1.0389445, 1.0297769
4: -4.8394833, -3.3299661, -4.7673016, -3.3324771, -1.4140642, 1.3905447
5: -8.3733959, -6.7834020, -8.3700256, -6.7942047, -1.2155111, 1.2211022
6: -6.0029597, -3.9402688, -5.9793053, -3.9414666, -1.6773057, 1.6585405
7: -4.2085056, -2.7925813, -4.2065663, -2.8140085, -1.3558991, 1.3753953
8: -3.7436275, -2.2970471, -3.7371254, -2.3019571, -1.1623292, 1.1568391
9: -11.0767317, -9.1578217, -11.0496931, -9.1602621, -1.4784446, 1.4570837

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7327093, upper bound: 0.7192858
time: 3.71 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464879, upper bound: 0.7464874
time: 3.73 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.8857450, -6.7203894, -8.8104820, -6.7428751, -1.7234993, 1.7184801
1: 1.9390879, 3.6986580, 1.9664164, 3.6431956, -1.3789256, 1.3762670
2: -5.4787116, -3.7883573, -5.4576225, -3.8527575, -1.3284218, 1.3141830
3: -10.1732092, -8.3885546, -10.1618958, -8.4089861, -1.0209731, 1.0380383
4: -4.7905512, -3.2545400, -4.7664204, -3.3353643, -1.4104345, 1.3942735
5: -8.3966160, -6.7893448, -8.3696098, -6.7945609, -1.2121141, 1.2161465
6: -5.9908571, -3.9333987, -5.9789581, -3.9421825, -1.6617289, 1.6639930
7: -4.2135053, -2.7859933, -4.2058878, -2.8142688, -1.3599923, 1.3459630
8: -3.7674494, -2.2902737, -3.7366943, -2.3020444, -1.1844602, 1.1659064
9: -11.1725769, -9.1324453, -11.0492449, -9.1611757, -1.4761584, 1.4816973

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528341, upper bound: 0.7438240
time: 3.72 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528341, upper bound: 0.7464665
time: 3.72 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.9012508, -6.6729517, -8.8123455, -6.7421470, -1.7205868, 1.7451353
1: 1.9283442, 3.7209811, 1.9655838, 3.6435561, -1.3922064, 1.3763156
2: -5.4885020, -3.7852988, -5.4579525, -3.8526151, -1.3463202, 1.3259082
3: -10.1965437, -8.3811646, -10.1621256, -8.4082718, -1.0507724, 1.0451324
4: -4.8682165, -3.2327337, -4.7673016, -3.3324771, -1.4430587, 1.4167466
5: -8.4027042, -6.7760482, -8.3700256, -6.7942047, -1.2352667, 1.2287037
6: -6.0169468, -3.9272974, -5.9793053, -3.9414666, -1.6925254, 1.6710427
7: -4.2200108, -2.7628975, -4.2065663, -2.8140085, -1.3675973, 1.3859856
8: -3.7768860, -2.2847967, -3.7371254, -2.3019571, -1.1812901, 1.1727663
9: -11.2026215, -9.1239042, -11.0496931, -9.1602621, -1.4828739, 1.4900427

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283480, upper bound: 0.7326942
time: 3.81 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554878, upper bound: 0.7464798
time: 3.66 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.8010015, -6.7331152, -8.8857450, -6.7203894, -1.7327278, 1.7326198
1: 1.9590774, 3.6416245, 1.9390879, 3.6986580, -1.3835363, 1.3862672
2: -5.4673090, -3.8522122, -5.4787116, -3.7883573, -1.3216560, 1.3194599
3: -10.1634703, -8.4114809, -10.1732092, -8.3885546, -1.0245850, 1.0257903
4: -4.7789888, -3.3510437, -4.7905512, -3.2545400, -1.4093349, 1.4021599
5: -8.3706951, -6.7922735, -8.3966160, -6.7893448, -1.2094994, 1.2156508
6: -5.9808726, -3.9460056, -5.9908571, -3.9333987, -1.6570451, 1.6581116
7: -4.2049284, -2.8143258, -4.2135053, -2.7859933, -1.3478332, 1.3511069
8: -3.7358589, -2.2974715, -3.7674494, -2.2902737, -1.1894403, 1.1894891
9: -11.0471544, -9.1433182, -11.1725769, -9.1324453, -1.4857192, 1.4940009

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7528352
time: 3.91 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7439057
time: 3.88 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.8165398, -6.6856232, -8.8857450, -6.7203894, -1.7280207, 1.7420459
1: 1.9484758, 3.6639700, 1.9390879, 3.6986580, -1.3956480, 1.3991454
2: -5.4771199, -3.8491411, -5.4787116, -3.7883573, -1.3298948, 1.3314066
3: -10.1867418, -8.4040794, -10.1732092, -8.3885546, -1.0478246, 1.0322256
4: -4.8566618, -3.3291776, -4.7905512, -3.2545400, -1.4205880, 1.4242389
5: -8.3768263, -6.7789831, -8.3966160, -6.7893448, -1.2318604, 1.2199936
6: -6.0068855, -3.9399207, -5.9908571, -3.9333987, -1.6805286, 1.6652744
7: -4.2114773, -2.7911191, -4.2135053, -2.7859933, -1.3546338, 1.3900304
8: -3.7451873, -2.2919660, -3.7674494, -2.2902737, -1.1844629, 1.1932068
9: -11.0772896, -9.1347513, -11.1725769, -9.1324453, -1.5078773, 1.5013162

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7528337
time: 3.64 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7439046
time: 4.15 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.8123455, -6.7421470, -8.9012508, -6.6729517, -1.7451353, 1.7205870
1: 1.9655838, 3.6435561, 1.9283442, 3.7209811, -1.3763156, 1.3922064
2: -5.4579525, -3.8526151, -5.4885020, -3.7852988, -1.3259079, 1.3463199
3: -10.1621256, -8.4082718, -10.1965437, -8.3811646, -1.0451324, 1.0507724
4: -4.7673016, -3.3324771, -4.8682165, -3.2327337, -1.4167466, 1.4430585
5: -8.3700256, -6.7942047, -8.4027042, -6.7760482, -1.2287037, 1.2352666
6: -5.9793053, -3.9414666, -6.0169468, -3.9272974, -1.6710424, 1.6925254
7: -4.2065663, -2.8140085, -4.2200108, -2.7628975, -1.3859859, 1.3675973
8: -3.7371254, -2.3019571, -3.7768860, -2.2847967, -1.1727664, 1.1812901
9: -11.0496931, -9.1602621, -11.2026215, -9.1239042, -1.4900427, 1.4828738

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7326952, upper bound: 0.7283476
time: 3.60 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464790, upper bound: 0.7554899
time: 3.70 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8981571, -6.7153130, -8.9012508, -6.6729517, -1.7503109, 1.7458043
1: 1.9334898, 3.7011232, 1.9283442, 3.7209811, -1.4066339, 1.4072926
2: -5.4809976, -3.7874246, -5.4885020, -3.7852988, -1.3488619, 1.3538613
3: -10.1747761, -8.3837395, -10.1965437, -8.3811646, -1.0582268, 1.0615385
4: -4.7960482, -3.2351804, -4.8682165, -3.2327337, -1.4394653, 1.4518971
5: -8.3994341, -6.7868547, -8.4027042, -6.7760482, -1.2427926, 1.2415359
6: -5.9932270, -3.9285104, -6.0169468, -3.9272974, -1.6844316, 1.6986289
7: -4.2181134, -2.7842031, -4.2200108, -2.7628975, -1.4025912, 1.3986416
8: -3.7703381, -2.2896957, -3.7768860, -2.2847967, -1.1910629, 1.1967232
9: -11.1756630, -9.1263142, -11.2026215, -9.1239042, -1.5061545, 1.5161738

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7326952, upper bound: 0.7192741
time: 3.89 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464790, upper bound: 0.7192722
time: 9.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.99 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7438337, upper bound: 0.7438328
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7438337, upper bound: 0.7464753
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7327093, upper bound: 0.7192858
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7464879, upper bound: 0.7464874
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7528341, upper bound: 0.7438240
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7528341, upper bound: 0.7464665
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7283480, upper bound: 0.7326942
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7554878, upper bound: 0.7464798
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7528352
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7439057
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7528337
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7438270, upper bound: 0.7439046
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7326952, upper bound: 0.7283476
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7464790, upper bound: 0.7554899
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7326952, upper bound: 0.7192741
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 1, lower bound: -0.7464790, upper bound: 0.7192722

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472324, -8.7999287, -6.7472324, -1.7011480, 1.7011480
1: 1.9711514, 3.6410933, 1.9711514, 3.6410933, -1.3548231, 1.3548232
2: -5.4556670, -3.8535542, -5.4556670, -3.8535542, -1.2943685, 1.2943685
3: -10.1605568, -8.4130840, -10.1605568, -8.4130840, -1.0050344, 1.0050343
4: -4.7618060, -3.3518276, -4.7618060, -3.3518276, -1.3652430, 1.3652430
5: -8.3672218, -6.7966924, -8.3672218, -6.7966924, -1.1913404, 1.1913406
6: -5.9769564, -3.9463489, -5.9769564, -3.9463489, -1.6414659, 1.6414661
7: -4.2019610, -2.8157861, -4.2019610, -2.8157861, -1.3187618, 1.3187618
8: -3.7342720, -2.3025498, -3.7342720, -2.3025498, -1.1617855, 1.1617854
9: -11.0466022, -9.1663828, -11.0466022, -9.1663828, -1.4502316, 1.4502318

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7165514, upper bound: 0.7300542
time: 3.79 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438296, upper bound: 0.7438325
time: 3.85 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472324, -8.8154755, -6.6997294, -1.7229934, 1.6964650
1: 1.9711514, 3.6410933, 1.9605536, 3.6634364, -1.3677015, 1.3675435
2: -5.4556670, -3.8535542, -5.4654775, -3.8504770, -1.3064106, 1.3042452
3: -10.1605568, -8.4130840, -10.1837931, -8.4056721, -1.0121744, 1.0340619
4: -4.7618060, -3.3518276, -4.8394814, -3.3299668, -1.3873231, 1.3945928
5: -8.3672218, -6.7966924, -8.3733950, -6.7834015, -1.2029393, 1.2137105
6: -5.9769564, -3.9463489, -6.0029583, -3.9402688, -1.6486251, 1.6714528
7: -4.2019610, -2.8157861, -4.2085056, -2.7925818, -1.3698478, 1.3262658
8: -3.7342720, -2.3025498, -3.7436275, -2.2970462, -1.1681666, 1.1569674
9: -11.0466022, -9.1663828, -11.0767307, -9.1578197, -1.4584394, 1.4720907

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7300537, upper bound: 0.7192134
time: 3.86 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438307, upper bound: 0.7464723
time: 3.76 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -8.7900629, -6.7154074, -8.8045473, -6.7483516, -1.6717272, 1.6627488
1: 2.0048685, 3.5934381, 1.9789362, 3.6228075, -1.2295179, 1.2896397
2: -5.4552307, -3.8929613, -5.4538488, -3.8657589, -1.2482142, 1.2613986
3: -10.1735210, -8.4316940, -10.1588783, -8.4245338, -1.0158386, 0.9988853
4: -4.7823601, -3.3369749, -4.7444906, -3.3346212, -1.3568847, 1.3459041
5: -8.2931137, -6.7997208, -8.3446426, -6.7995825, -1.1252205, 1.1412370
6: -5.9411144, -4.0652766, -5.9604998, -3.9777899, -1.4782813, 1.5167216
7: -4.1293292, -2.8156500, -4.1834641, -2.8222044, -1.2719431, 1.1748171
8: -3.7338781, -2.3167763, -3.7323246, -2.3082161, -1.1449207, 1.1285527
9: -10.9417534, -9.1931400, -11.0101957, -9.1719761, -1.3405659, 1.2113721

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192865, upper bound: 0.7192863
time: 3.83 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192865, upper bound: 0.7192868
time: 5.01 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.8154745, -6.6997290, -8.8123455, -6.7421470, -1.7016315, 1.7672324
1: 1.9605546, 3.6634345, 1.9655838, 3.6435561, -1.3840427, 1.3742492
2: -5.4654770, -3.8504791, -5.4579525, -3.8526151, -1.3265493, 1.3105125
3: -10.1837902, -8.4056749, -10.1621256, -8.4082718, -1.0381444, 1.0648179
4: -4.8394790, -3.3299670, -4.7673016, -3.3324771, -1.4499230, 1.3879101
5: -8.3733931, -6.7834034, -8.3700256, -6.7942047, -1.2092736, 1.2177587
6: -6.0029569, -3.9402728, -5.9793053, -3.9414666, -1.6900535, 1.6585367
7: -4.2085009, -2.7925828, -4.2065663, -2.8140085, -1.3558943, 1.4105847
8: -3.7436285, -2.2970476, -3.7371254, -2.3019571, -1.1644214, 1.1568382
9: -11.0767269, -9.1578236, -11.0496931, -9.1602621, -1.4784439, 1.4634342

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B1_A1_A2_A2_A1

### Relational analysis result of IS_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7423120, upper bound: 0.7451323
time: 3.73 seconds

## Relational analysis of IS_B1_A1_A2_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464824, upper bound: 0.7464820
time: 3.79 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.8857450, -6.7203894, -8.7999287, -6.7472324, -1.7182765, 1.7275219
1: 1.9390879, 3.6986580, 1.9711514, 3.6410933, -1.3850033, 1.3708303
2: -5.4787116, -3.7883573, -5.4556670, -3.8535542, -1.3175459, 1.3101437
3: -10.1732092, -8.3885546, -10.1605568, -8.4130840, -1.0168090, 1.0201111
4: -4.7905512, -3.2545400, -4.7618060, -3.3518276, -1.3939345, 1.3912413
5: -8.3966160, -6.7893448, -8.3672218, -6.7966924, -1.2106802, 1.1988771
6: -5.9908571, -3.9333987, -5.9769564, -3.9463489, -1.6567729, 1.6539454
7: -4.2135053, -2.7859933, -4.2019610, -2.8157861, -1.3304892, 1.3413415
8: -3.7674494, -2.2902737, -3.7342720, -2.3025498, -1.1834950, 1.1779032
9: -11.1725769, -9.1324453, -11.0466022, -9.1663828, -1.4709241, 1.4831698

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7256128, upper bound: 0.7300398
time: 3.88 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528297, upper bound: 0.7438238
time: 3.78 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.8857450, -6.7203894, -8.8154755, -6.6997294, -1.7277117, 1.7227609
1: 1.9390879, 3.6986580, 1.9605536, 3.6634364, -1.3979254, 1.3829563
2: -5.4787116, -3.7883573, -5.4654775, -3.8504770, -1.3294828, 1.3183644
3: -10.1732092, -8.3885546, -10.1837931, -8.4056721, -1.0239494, 1.0433543
4: -4.7905512, -3.2545400, -4.8394814, -3.3299668, -1.4160149, 1.4025006
5: -8.3966160, -6.7893448, -8.3733950, -6.7834015, -1.2150245, 1.2212536
6: -5.9908571, -3.9333987, -6.0029583, -3.9402688, -1.6639316, 1.6774485
7: -4.2135053, -2.7859933, -4.2085056, -2.7925818, -1.3816833, 1.3481302
8: -3.7674494, -2.2902737, -3.7436275, -2.2970462, -1.1872196, 1.1729449
9: -11.1725769, -9.1324453, -11.0767307, -9.1578197, -1.4782343, 1.5054135

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7256128, upper bound: 0.7326791
time: 3.80 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7528297, upper bound: 0.7464648
time: 3.81 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8934822, -6.6821985, -8.7869253, -6.7579765, -1.6640940, 1.7147632
1: 1.9436712, 3.7000551, 2.0097251, 3.5735221, -1.3063639, 1.2308178
2: -5.4833260, -3.7986007, -5.4477673, -3.8951230, -1.2966838, 1.2476482
3: -10.1930723, -8.4017105, -10.1518898, -8.4342422, -1.0170717, 1.0189756
4: -4.8430276, -3.2348907, -4.7095113, -3.3394938, -1.3952808, 1.3595455
5: -8.3771820, -6.7823954, -8.2896385, -6.8105416, -1.1545820, 1.1379790
6: -5.9953237, -3.9636536, -5.9173508, -4.0664763, -1.5477672, 1.4792981
7: -4.1968470, -2.7762551, -4.1273603, -2.8371060, -1.1822565, 1.3009129
8: -3.7703753, -2.2913251, -3.7275162, -2.3217225, -1.1519833, 1.1543598
9: -11.1629562, -9.1378937, -10.9146261, -9.1954803, -1.2209044, 1.3523617

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283480, upper bound: 0.7192739
time: 3.97 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283480, upper bound: 0.7326941
time: 3.84 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9012508, -6.6729517, -8.8123455, -6.7421484, -1.7690783, 1.7451344
1: 1.9283442, 3.7209811, 1.9655857, 3.6435533, -1.3922029, 1.3914258
2: -5.4885020, -3.7852988, -5.4579520, -3.8526168, -1.3463178, 1.3277018
3: -10.1965437, -8.3811646, -10.1621256, -8.4082747, -1.0833144, 1.0443351
4: -4.8682165, -3.2327337, -4.7672987, -3.3324771, -1.4404187, 1.4525528
5: -8.4027042, -6.7760482, -8.3700237, -6.7942042, -1.2319317, 1.2224655
6: -6.0169468, -3.9272974, -5.9793038, -3.9414718, -1.6925244, 1.6900771
7: -4.2200108, -2.7628975, -4.2065625, -2.8140092, -1.4060016, 1.3859851
8: -3.7768860, -2.2847967, -3.7371235, -2.3019609, -1.1812892, 1.1748583
9: -11.2026215, -9.1239042, -11.0496874, -9.1602650, -1.4778886, 1.4900377

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7513649, upper bound: 0.7451215
time: 3.89 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554822, upper bound: 0.7464746
time: 4.02 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472324, -8.8857450, -6.7203894, -1.7275217, 1.7182765
1: 1.9711514, 3.6410933, 1.9390879, 3.6986580, -1.3708301, 1.3850036
2: -5.4556670, -3.8535542, -5.4787116, -3.7883573, -1.3101437, 1.3175459
3: -10.1605568, -8.4130840, -10.1732092, -8.3885546, -1.0201108, 1.0168092
4: -4.7618060, -3.3518276, -4.7905512, -3.2545400, -1.3912413, 1.3939345
5: -8.3672218, -6.7966924, -8.3966160, -6.7893448, -1.1988771, 1.2106802
6: -5.9769564, -3.9463489, -5.9908571, -3.9333987, -1.6539457, 1.6567727
7: -4.2019610, -2.8157861, -4.2135053, -2.7859933, -1.3413415, 1.3304894
8: -3.7342720, -2.3025498, -3.7674494, -2.2902737, -1.1779032, 1.1834946
9: -11.0466022, -9.1663828, -11.1725769, -9.1324453, -1.4831696, 1.4709241

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7300409, upper bound: 0.7256136
time: 3.73 seconds

## Relational analysis of IS_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438231, upper bound: 0.7528333
time: 3.69 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.8857450, -6.7203894, -8.8857450, -6.7203894, -1.7452002, 1.7451999
1: 1.9390879, 3.6986580, 1.9390879, 3.6986580, -1.4010534, 1.4010534
2: -5.4787116, -3.7883573, -5.4787116, -3.7883573, -1.3331673, 1.3331671
3: -10.1732092, -8.3885546, -10.1732092, -8.3885546, -1.0332284, 1.0332285
4: -4.7905512, -3.2545400, -4.7905512, -3.2545400, -1.4141438, 1.4141440
5: -8.3966160, -6.7893448, -8.3966160, -6.7893448, -1.2171969, 1.2171969
6: -5.9908571, -3.9333987, -5.9908571, -3.9333987, -1.6673338, 1.6673338
7: -4.2135053, -2.7859933, -4.2135053, -2.7859933, -1.3578382, 1.3578384
8: -3.7674494, -2.2902737, -3.7674494, -2.2902737, -1.1962752, 1.1962756
9: -11.1725769, -9.1324453, -11.1725769, -9.1324453, -1.4992805, 1.4992805

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7300409, upper bound: 0.7165376
time: 5.88 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438231, upper bound: 0.7438225
time: 6.80 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.8154755, -6.6997294, -8.8857450, -6.7203894, -1.7227609, 1.7277117
1: 1.9605536, 3.6634364, 1.9390879, 3.6986580, -1.3829565, 1.3979249
2: -5.4654775, -3.8504770, -5.4787116, -3.7883573, -1.3183649, 1.3294828
3: -10.1837931, -8.4056721, -10.1732092, -8.3885546, -1.0433543, 1.0239491
4: -4.8394814, -3.3299668, -4.7905512, -3.2545400, -1.4025006, 1.4160149
5: -8.3733950, -6.7834015, -8.3966160, -6.7893448, -1.2212536, 1.2150245
6: -6.0029583, -3.9402688, -5.9908571, -3.9333987, -1.6774487, 1.6639315
7: -4.2085056, -2.7925818, -4.2135053, -2.7859933, -1.3481302, 1.3816833
8: -3.7436275, -2.2970462, -3.7674494, -2.2902737, -1.1729450, 1.1872194
9: -11.0767307, -9.1578197, -11.1725769, -9.1324453, -1.5054135, 1.4782343

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7326799, upper bound: 0.7256123
time: 3.83 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464637, upper bound: 0.7528318
time: 3.84 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.9012518, -6.6729546, -8.8857450, -6.7203894, -1.7404532, 1.7546387
1: 1.9283447, 3.7209816, 1.9390879, 3.6986580, -1.4133482, 1.3999906
2: -5.4885006, -3.7852983, -5.4787116, -3.7883573, -1.3415449, 1.3447061
3: -10.1965427, -8.3811674, -10.1732092, -8.3885546, -1.0566375, 1.0396858
4: -4.8682165, -3.2327342, -4.7905512, -3.2545400, -1.4324222, 1.4362171
5: -8.4027052, -6.7760496, -8.3966160, -6.7893448, -1.2397308, 1.2246013
6: -6.0169468, -3.9272974, -5.9908571, -3.9333987, -1.6927762, 1.6745157
7: -4.2200108, -2.7628984, -4.2135053, -2.7859933, -1.3646083, 1.3971794
8: -3.7768860, -2.2847958, -3.7674494, -2.2902737, -1.1914529, 1.2025933
9: -11.2026205, -9.1239052, -11.1725769, -9.1324453, -1.5098431, 1.5075090

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192018, upper bound: 0.7300383
time: 4.22 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464626, upper bound: 0.7438220
time: 5.87 seconds

## BFS IS instance: IS_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -8.7869253, -6.7579765, -8.8934822, -6.6821985, -1.7147632, 1.6640940
1: 2.0097251, 3.5735221, 1.9436712, 3.7000551, -1.2308178, 1.3063638
2: -5.4477673, -3.8951230, -5.4833260, -3.7986007, -1.2476482, 1.2966838
3: -10.1518898, -8.4342422, -10.1930723, -8.4017105, -1.0189757, 1.0170717
4: -4.7095113, -3.3394938, -4.8430276, -3.2348907, -1.3595452, 1.3952808
5: -8.2896385, -6.8105416, -8.3771820, -6.7823954, -1.1379790, 1.1545820
6: -5.9173508, -4.0664763, -5.9953237, -3.9636536, -1.4792984, 1.5477674
7: -4.1273603, -2.8371060, -4.1968470, -2.7762551, -1.3009129, 1.1822565
8: -3.7275162, -2.3217225, -3.7703753, -2.2913251, -1.1543596, 1.1519833
9: -10.9146261, -9.1954803, -11.1629562, -9.1378937, -1.3523617, 1.2209045

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B2_B2_A1_A1_B1

### Relational analysis result of IS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192745, upper bound: 0.7283474
time: 3.70 seconds

## Relational analysis of IS_B2_B2_A1_A1_B2

### Relational analysis result of IS_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7192745, upper bound: 0.7283480
time: 4.52 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.8123455, -6.7421484, -8.9012508, -6.6729517, -1.7451344, 1.7690783
1: 1.9655857, 3.6435533, 1.9283442, 3.7209811, -1.3914256, 1.3922029
2: -5.4579520, -3.8526168, -5.4885020, -3.7852988, -1.3277018, 1.3463180
3: -10.1621256, -8.4082747, -10.1965437, -8.3811646, -1.0443351, 1.0833144
4: -4.7672987, -3.3324771, -4.8682165, -3.2327337, -1.4525526, 1.4404190
5: -8.3700237, -6.7942042, -8.4027042, -6.7760482, -1.2224655, 1.2319317
6: -5.9793038, -3.9414718, -6.0169468, -3.9272974, -1.6900773, 1.6925244
7: -4.2065625, -2.8140092, -4.2200108, -2.7628975, -1.3859849, 1.4060016
8: -3.7371235, -2.3019609, -3.7768860, -2.2847967, -1.1748583, 1.1812892
9: -11.0496874, -9.1602650, -11.2026215, -9.1239042, -1.4900374, 1.4778885

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7451220, upper bound: 0.7513657
time: 3.71 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7464737, upper bound: 0.7554821
time: 3.98 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.8730726, -6.7341552, -8.8936806, -6.6812677, -1.7201271, 1.6891804
1: 1.9800944, 3.6309047, 1.9412479, 3.7003889, -1.2596941, 1.3229051
2: -5.4694271, -3.8297954, -5.4848089, -3.7981350, -1.2715626, 1.3025184
3: -10.1643581, -8.4138212, -10.1933823, -8.3971176, -1.0325952, 1.0236377
4: -4.7349868, -3.2421246, -4.8458228, -3.2347593, -1.3778529, 1.4054792
5: -8.3198757, -6.8043160, -8.3782749, -6.7809725, -1.1509998, 1.1653421
6: -5.9275494, -4.0536261, -5.9988680, -3.9636438, -1.4924307, 1.5580754
7: -4.1388445, -2.8128128, -4.1969857, -2.7723496, -1.3185749, 1.1910686
8: -3.7589488, -2.3098335, -3.7722640, -2.2909746, -1.1705430, 1.1688093
9: -11.0403757, -9.1633596, -11.1632681, -9.1340752, -1.3683863, 1.2601293

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of IS_B2_B2_A2_A1_B1

### Relational analysis result of IS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283479, upper bound: 0.7192736
time: 3.95 seconds

## Relational analysis of IS_B2_B2_A2_A1_B2

### Relational analysis result of IS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7283479, upper bound: 0.7192712
time: 4.44 seconds

## BFS IS instance: IS_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.8981533, -6.7153139, -8.9012508, -6.6729517, -1.7503099, 1.7961535
1: 1.9334908, 3.7011204, 1.9283442, 3.7209811, -1.4220910, 1.4072921
2: -5.4809966, -3.7874269, -5.4885020, -3.7852988, -1.3508179, 1.3538599
3: -10.1747751, -8.3837423, -10.1965437, -8.3811646, -1.0575321, 1.0926964
4: -4.7960453, -3.2351799, -4.8682165, -3.2327337, -1.4764757, 1.4492989
5: -8.3994312, -6.7868567, -8.4027042, -6.7760482, -1.2365530, 1.2415353
6: -5.9932227, -3.9285150, -6.0169468, -3.9272974, -1.7058935, 1.6986287
7: -4.2181067, -2.7842052, -4.2200108, -2.7628975, -1.4025908, 1.4329226
8: -3.7703352, -2.2896967, -3.7768860, -2.2847967, -1.1931559, 1.1967216
9: -11.1756554, -9.1263161, -11.2026215, -9.1239042, -1.5061498, 1.5117707

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A2_A2_B1

### Relational analysis result of IS_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7541861, upper bound: 0.7423604
time: 3.73 seconds

## Relational analysis of IS_B2_B2_A2_A2_B2

### Relational analysis result of IS_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554834, upper bound: 0.7465483
time: 4.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.80 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7165514, upper bound: 0.7300542
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7438296, upper bound: 0.7438325
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7300537, upper bound: 0.7192134
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7438307, upper bound: 0.7464723
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7192865, upper bound: 0.7192863
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7192865, upper bound: 0.7192868
IS_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7423120, upper bound: 0.7451323
IS_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7464824, upper bound: 0.7464820
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7256128, upper bound: 0.7300398
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7528297, upper bound: 0.7438238
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7256128, upper bound: 0.7326791
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7528297, upper bound: 0.7464648
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7283480, upper bound: 0.7192739
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7283480, upper bound: 0.7326941
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7513649, upper bound: 0.7451215
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7554822, upper bound: 0.7464746
IS_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7300409, upper bound: 0.7256136
IS_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7438231, upper bound: 0.7528333
IS_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7300409, upper bound: 0.7165376
IS_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7438231, upper bound: 0.7438225
IS_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7326799, upper bound: 0.7256123
IS_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7464637, upper bound: 0.7528318
IS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7192018, upper bound: 0.7300383
IS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7464626, upper bound: 0.7438220
IS_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7192745, upper bound: 0.7283474
IS_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7192745, upper bound: 0.7283480
IS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7451220, upper bound: 0.7513657
IS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7464737, upper bound: 0.7554821
IS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7283479, upper bound: 0.7192736
IS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7283479, upper bound: 0.7192712
IS_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7541861, upper bound: 0.7423604
IS_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 1, lower bound: -0.7554834, upper bound: 0.7465483

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.7921257, -6.7534146, -8.7744923, -6.7630730, -1.6509004, 1.6720660
1: 1.9844952, 3.6203485, 2.0152016, 3.5710583, -1.2705396, 1.2236052
2: -5.4515724, -3.8666902, -5.4455094, -3.8960195, -1.2452409, 1.2226534
3: -10.1573143, -8.4293528, -10.1503220, -8.4390316, -0.9742577, 0.9817103
4: -4.7391605, -3.3539805, -4.7044754, -3.3588679, -1.3206236, 1.3081026
5: -8.3418169, -6.8020864, -8.2867794, -6.8130946, -1.1190214, 1.1007543
6: -5.9581299, -3.9826741, -5.9148703, -4.0713606, -1.4996376, 1.4558563
7: -4.1788597, -2.8239760, -4.1227598, -2.8388700, -1.1311252, 1.2348096
8: -3.7294731, -2.3088069, -3.7246737, -2.3223124, -1.1333780, 1.1444148
9: -11.0071096, -9.1780977, -10.9115448, -9.2015972, -1.2047813, 1.3129091

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7165528, upper bound: 0.7165523
time: 3.67 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7165528, upper bound: 0.7300543
time: 3.60 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.7999287, -6.7472324, -8.7999287, -6.7472343, -1.7540369, 1.7011454
1: 1.9711514, 3.6410933, 1.9711537, 3.6410923, -1.3548198, 1.3770057
2: -5.4556670, -3.8535542, -5.4556665, -3.8535562, -1.2943656, 1.2984984
3: -10.1605568, -8.4130840, -10.1605568, -8.4130869, -1.0401487, 1.0042298
4: -4.7618060, -3.3518276, -4.7618022, -3.3518291, -1.3626060, 1.4039507
5: -8.3672218, -6.7966924, -8.3672190, -6.7966933, -1.1913404, 1.1851041
6: -5.9769564, -3.9463489, -5.9769545, -3.9463549, -1.6414623, 1.6644154
7: -4.2019610, -2.8157861, -4.2019572, -2.8157864, -1.3618760, 1.3187582
8: -3.7342720, -2.3025498, -3.7342701, -2.3025527, -1.1617844, 1.1638782
9: -11.0466022, -9.1663828, -11.0465975, -9.1663837, -1.4565818, 1.4502265

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of IS_B1_A1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7424864, upper bound: 0.7396668
time: 3.90 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7438259, upper bound: 0.7438267
time: 3.86 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7744923, -6.7630730, -8.8076820, -6.7058210, -1.6938658, 1.6456399
1: 2.0152016, 3.5710583, 1.9739933, 3.6426973, -1.2225914, 1.2832036
2: -5.4455094, -3.8960195, -5.4613523, -3.8636024, -1.2347395, 1.2550733
3: -10.1503220, -8.4390316, -10.1805305, -8.4219208, -0.9888933, 1.0012565
4: -4.7044754, -3.3588679, -4.8168344, -3.3321357, -1.3301933, 1.3499107
5: -8.2867794, -6.8130946, -8.3480692, -6.7887821, -1.1129742, 1.1412838
6: -5.9148703, -4.0713606, -5.9841795, -3.9765937, -1.4623477, 1.5302074
7: -4.1227598, -2.8388700, -4.1854143, -2.8007884, -1.2858925, 1.1378844
8: -3.7246737, -2.3223124, -3.7387247, -2.3032961, -1.1508389, 1.1286023
9: -10.9115448, -9.2015972, -11.0372581, -9.1695604, -1.3210900, 1.2106649

Time for backsubstitution: 14.77 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3692495822906494
rel_dist={1: [-0.7555267226325175, 0.7555256824713692]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5748
type: A, layer: 1, pos: 5748
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5748

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6733577, upper bound: 0.6660921
time: 3.42 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734512, upper bound: 0.6734513
time: 3.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.94 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 1, lower bound: -0.6733577, upper bound: 0.6660921
IS_B2, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 1, lower bound: -0.6734512, upper bound: 0.6734513

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.8132877, -6.7303009, -8.8123894, -6.7421303, -1.6394629, 1.6504505
1: 1.9554653, 3.6440086, 1.9655743, 3.6435642, -1.3198190, 1.3098437
2: -5.4677148, -3.8514876, -5.4579616, -3.8526118, -1.2870600, 1.2784903
3: -10.1645718, -8.4069157, -10.1621323, -8.4082584, -0.9643013, 0.9621580
4: -4.7817125, -3.3317838, -4.7673235, -3.3324428, -1.3546939, 1.3402495
5: -8.3729439, -6.7904968, -8.3700380, -6.7941961, -1.1588397, 1.1583042
6: -5.9825916, -3.9411530, -5.9793115, -3.9414420, -1.6028214, 1.6007271
7: -4.2090664, -2.8127747, -4.2065864, -2.8140006, -1.3274555, 1.3246417
8: -3.7384624, -2.2977057, -3.7371349, -2.3019557, -1.1027497, 1.1060107
9: -11.0501690, -9.1409216, -11.0497036, -9.1602478, -1.3962688, 1.4149337

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6732361, upper bound: 0.6639738
time: 3.87 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6733548, upper bound: 0.6660891
time: 3.48 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.8134594, -6.7280216, -8.8982000, -6.7152944, -1.6664777, 1.6713221
1: 1.9535041, 3.6440935, 1.9334784, 3.7011304, -1.3364904, 1.3390954
2: -5.4695988, -3.8512712, -5.4810061, -3.7874212, -1.3028910, 1.3003247
3: -10.1650486, -8.4066563, -10.1747808, -8.3837261, -0.9797161, 0.9801023
4: -4.7844954, -3.3316591, -4.7960691, -3.2351460, -1.3830168, 1.3721960
5: -8.3735027, -6.7897782, -8.3994455, -6.7868471, -1.1710799, 1.1779418
6: -5.9832315, -3.9410977, -5.9932337, -3.9284849, -1.6158133, 1.6167415
7: -4.2095542, -2.8125379, -4.2181315, -2.7841961, -1.3532336, 1.3551335
8: -3.7387180, -2.2968798, -3.7703476, -2.2896943, -1.1273991, 1.1283591
9: -11.0502615, -9.1371880, -11.1756716, -9.1263027, -1.4255652, 1.4382175

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6733316, upper bound: 0.6713343
time: 3.55 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734483, upper bound: 0.6734483
time: 3.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.88 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 1, lower bound: -0.6732361, upper bound: 0.6639738
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 1, lower bound: -0.6733548, upper bound: 0.6660891
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 1, lower bound: -0.6733316, upper bound: 0.6713343
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 1, lower bound: -0.6734483, upper bound: 0.6734483

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -8.8090630, -6.7319636, -8.7999287, -6.7472324, -1.6299603, 1.6569953
1: 1.9573293, 3.6431847, 1.9711514, 3.6410933, -1.3231466, 1.3019602
2: -5.4669600, -3.8518081, -5.4556670, -3.8535542, -1.2734768, 1.2734468
3: -10.1640444, -8.4085264, -10.1605568, -8.4130840, -0.9584842, 0.9421970
4: -4.7796988, -3.3382633, -4.7618060, -3.3518276, -1.3347416, 1.3304417
5: -8.3719921, -6.7913127, -8.3672218, -6.7966924, -1.1557631, 1.1402494
6: -5.9818020, -3.9427958, -5.9769564, -3.9463489, -1.5962420, 1.5881793
7: -4.2075171, -2.8133700, -4.2019610, -2.8157861, -1.2920961, 1.3180273
8: -3.7374840, -2.2979012, -3.7342720, -2.3025498, -1.0997674, 1.1160561
9: -11.0491467, -9.1429787, -11.0466022, -9.1663828, -1.3890564, 1.4139323

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6659757, upper bound: 0.6639740
time: 3.98 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6659757, upper bound: 0.6639732
time: 3.82 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -8.8132343, -6.7303209, -8.8154755, -6.6997294, -1.6583600, 1.6527634
1: 1.9554763, 3.6439991, 1.9605546, 3.6634359, -1.3375831, 1.3156002
2: -5.4677038, -3.8514915, -5.4654770, -3.8504767, -1.2876859, 1.2914970
3: -10.1645651, -8.4069309, -10.1837940, -8.4056721, -0.9655648, 0.9724723
4: -4.7816882, -3.3318245, -4.8394833, -3.3299661, -1.3572030, 1.3657279
5: -8.3729286, -6.7905073, -8.3733959, -6.7834020, -1.1680806, 1.1626184
6: -5.9825826, -3.9411821, -6.0029597, -3.9402688, -1.6031504, 1.6191974
7: -4.2090435, -2.8127835, -4.2085056, -2.7925813, -1.3484311, 1.3260882
8: -3.7384496, -2.2977076, -3.7436275, -2.2970471, -1.1074430, 1.1159652
9: -11.0501585, -9.1409378, -11.0767317, -9.1578217, -1.3975685, 1.4362580

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6660920, upper bound: 0.6660871
time: 3.57 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6660920, upper bound: 0.6660881
time: 5.45 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -8.8092365, -6.7296829, -8.8857450, -6.7203894, -1.6569576, 1.6760683
1: 1.9553680, 3.6432719, 1.9390879, 3.6986580, -1.3405819, 1.3311795
2: -5.4688449, -3.8515923, -5.4787116, -3.7883573, -1.2894504, 1.2952785
3: -10.1645212, -8.4082661, -10.1732092, -8.3885546, -0.9739642, 0.9597166
4: -4.7824812, -3.3381376, -4.7905512, -3.2545400, -1.3630271, 1.3623600
5: -8.3725529, -6.7905931, -8.3966160, -6.7893448, -1.1679997, 1.1595716
6: -5.9824409, -3.9427404, -5.9908571, -3.9333987, -1.6092293, 1.6041925
7: -4.2080050, -2.8131342, -4.2135053, -2.7859933, -1.3140044, 1.3486488
8: -3.7377386, -2.2970753, -3.7674494, -2.2902737, -1.1243781, 1.1381037
9: -11.0492373, -9.1392431, -11.1725769, -9.1324453, -1.4183769, 1.4370149

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6713341, upper bound: 0.6713343
time: 3.42 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6713341, upper bound: 0.6713329
time: 6.20 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -8.8134069, -6.7280407, -8.9012508, -6.6729517, -1.6829333, 1.6736968
1: 1.9535160, 3.6440849, 1.9283442, 3.7209811, -1.3419452, 1.3449850
2: -5.4695888, -3.8512745, -5.4885020, -3.7852988, -1.3035309, 1.3133585
3: -10.1650410, -8.4066706, -10.1965437, -8.3811646, -0.9810135, 0.9852320
4: -4.7844710, -3.3316994, -4.8682165, -3.2327337, -1.3857102, 1.3928018
5: -8.3734894, -6.7897882, -8.4027042, -6.7760482, -1.1765091, 1.1823783
6: -5.9832225, -3.9411271, -6.0169468, -3.9272974, -1.6161599, 1.6345303
7: -4.2095313, -2.8125460, -4.2200108, -2.7628975, -1.3588543, 1.3565662
8: -3.7387047, -2.2968817, -3.7768860, -2.2847967, -1.1319954, 1.1352844
9: -11.0502501, -9.1372032, -11.2026215, -9.1239042, -1.4269099, 1.4449637

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5748
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5748

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6660880, upper bound: 0.6733531
time: 3.51 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6660880, upper bound: 0.6663855
time: 6.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.17 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6659757, upper bound: 0.6639740
IS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6659757, upper bound: 0.6639732
IS_B1_B2_A1, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6660920, upper bound: 0.6660871
IS_B1_B2_A2, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6660920, upper bound: 0.6660881
IS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6713341, upper bound: 0.6713343
IS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6713341, upper bound: 0.6713329
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6660880, upper bound: 0.6733531
IS_B2_B2_A2, status: Status.VERIFIED, split count: 3, time: 25.17
Output dim: 1, lower bound: -0.6660880, upper bound: 0.6663855

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.8123379, -6.7421513, -8.9012508, -6.6729517, -1.6813622, 1.6593382
1: 1.9655862, 3.6435542, 1.9283442, 3.7209811, -1.3292298, 1.3453314
2: -5.4579506, -3.8526158, -5.4885020, -3.7852988, -1.2919819, 1.3117297
3: -10.1621246, -8.4082737, -10.1965437, -8.3811646, -0.9765511, 0.9823662
4: -4.7672977, -3.3324826, -4.8682165, -3.2327337, -1.3676138, 1.3911860
5: -8.3700247, -6.7942071, -8.4027042, -6.7760482, -1.1702626, 1.1773949
6: -5.9793034, -3.9414716, -6.0169468, -3.9272974, -1.6130562, 1.6338413
7: -4.2065630, -2.8140099, -4.2200108, -2.7628975, -1.3523712, 1.3352649
8: -3.7371230, -2.3019581, -3.7768860, -2.2847967, -1.1216466, 1.1292999
9: -11.0496922, -9.1602650, -11.2026215, -9.1239042, -1.4298749, 1.4218845

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6647883, upper bound: 0.6699504
time: 3.88 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6660840, upper bound: 0.6733488
time: 3.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.26 seconds
IS_B2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.26
Output dim: 1, lower bound: -0.6647883, upper bound: 0.6699504
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.26
Output dim: 1, lower bound: -0.6660840, upper bound: 0.6733488

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.8123341, -6.7421546, -8.9068031, -6.6695795, -1.6824970, 1.6594007
1: 1.9655962, 3.6435509, 1.9239883, 3.7284970, -1.3321385, 1.3468885
2: -5.4579248, -3.8526173, -5.4923272, -3.7638292, -1.2946703, 1.3083470
3: -10.1621237, -8.4082775, -10.1993256, -8.3772831, -0.9795938, 0.9827347
4: -4.7672892, -3.3324842, -4.8715501, -3.2271991, -1.3651025, 1.3891287
5: -8.3700218, -6.7942138, -8.4105816, -6.7740192, -1.1688592, 1.1795186
6: -5.9792995, -3.9414821, -6.0338488, -3.9245706, -1.6090045, 1.6370971
7: -4.2065496, -2.8140113, -4.2232819, -2.7516644, -1.3504498, 1.3321891
8: -3.7371159, -2.3019676, -3.7905440, -2.2832103, -1.1177536, 1.1286986
9: -11.0496855, -9.1602726, -11.2127533, -9.1222134, -1.4271235, 1.4241610

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of IS_B2_B2_A1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6516181, upper bound: 0.6486225
time: 4.10 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6660798, upper bound: 0.6733443
time: 3.50 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.42 seconds
IS_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.42
Output dim: 1, lower bound: -0.6516181, upper bound: 0.6486225
IS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.42
Output dim: 1, lower bound: -0.6660798, upper bound: 0.6733443

## BFS IS instance: IS_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8123312, -6.7421551, -8.9068031, -6.6695795, -1.6824961, 1.7058372
1: 1.9655986, 3.6435471, 1.9239883, 3.7284970, -1.3463864, 1.3468852
2: -5.4579239, -3.8526206, -5.4923272, -3.7638292, -1.2963035, 1.3083460
3: -10.1621227, -8.4082794, -10.1993256, -8.3772831, -0.9787965, 1.0136842
4: -4.7672858, -3.3324852, -4.8715501, -3.2271991, -1.3983982, 1.3864896
5: -8.3700190, -6.7942147, -8.4105816, -6.7740192, -1.1623087, 1.1761831
6: -5.9792967, -3.9414873, -6.0338488, -3.9245706, -1.6260405, 1.6370971
7: -4.2065449, -2.8140128, -4.2232819, -2.7516644, -1.3504488, 1.3736210
8: -3.7371168, -2.3019705, -3.7905440, -2.2832103, -1.1197646, 1.1286981
9: -11.0496798, -9.1602726, -11.2127533, -9.1222134, -1.4271188, 1.4189284

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of IS_B2_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6639640, upper bound: 0.6732254
time: 3.88 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6639640, upper bound: 0.6715103
time: 4.00 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.78 seconds
IS_B2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.78
Output dim: 1, lower bound: -0.6639640, upper bound: 0.6732254
IS_B2_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 22.78
Output dim: 1, lower bound: -0.6639640, upper bound: 0.6715103

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.7999249, -6.7472377, -8.9067831, -6.6703157, -1.6870289, 1.6968596
1: 1.9711657, 3.6410866, 1.9240394, 3.7283134, -1.3396435, 1.3527597
2: -5.4556408, -3.8535590, -5.4920473, -3.7638426, -1.2921433, 1.2900240
3: -10.1605549, -8.4130917, -10.1991768, -8.3773022, -0.9596660, 1.0087079
4: -4.7617941, -3.3518310, -4.8704977, -3.2272151, -1.3880477, 1.3659182
5: -8.3672180, -6.7967010, -8.4105492, -6.7741079, -1.1445034, 1.1737808
6: -5.9769502, -3.9463649, -6.0337806, -3.9245861, -1.6149030, 1.6311469
7: -4.2019434, -2.8157890, -4.2232685, -2.7517035, -1.3449681, 1.3405709
8: -3.7342653, -2.3025613, -3.7904797, -2.2832317, -1.1308688, 1.1265764
9: -11.0465927, -9.1663914, -11.2126026, -9.1222401, -1.4287829, 1.4124292

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6141
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6141

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6582900, upper bound: 0.6732158
time: 3.88 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6639543, upper bound: 0.6732157
time: 3.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 22.60 seconds
IS_B2_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6582900, upper bound: 0.6732158
IS_B2_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6639543, upper bound: 0.6732157

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -8.7970657, -6.8441854, -8.9066601, -6.6762810, -1.6047196, 1.6020963
1: 2.0266275, 3.6402521, 1.9276109, 3.7282829, -1.2820137, 1.3241355
2: -5.3638487, -3.8573408, -5.4860015, -3.7640285, -1.1993022, 1.2177374
3: -10.1247826, -8.4161263, -10.1969738, -8.3774672, -0.9232540, 0.9790471
4: -4.6537995, -3.3536658, -4.8633928, -3.2272732, -1.2796986, 1.2758398
5: -8.3609028, -6.8885627, -8.4102211, -6.7800083, -1.0575156, 1.0802728
6: -5.9694009, -3.9769111, -6.0331860, -3.9264724, -1.5850110, 1.6013949
7: -4.1942911, -2.8362992, -4.2228508, -2.7530949, -1.3245444, 1.3186407
8: -3.7321587, -2.3502336, -3.7903557, -2.2862582, -1.0965772, 1.0761950
9: -11.0431318, -9.2371178, -11.2124214, -9.1271677, -1.3789818, 1.3391757

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6554208, upper bound: 0.6728252
time: 3.68 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6582876, upper bound: 0.6732135
time: 3.91 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.7999239, -6.7472386, -8.9067831, -6.6703157, -1.6861620, 1.6262407
1: 1.9711661, 3.6410871, 1.9240394, 3.7283134, -1.3381259, 1.3527597
2: -5.4556417, -3.8535595, -5.4920473, -3.7638426, -1.2486010, 1.2900233
3: -10.1605530, -8.4130917, -10.1991768, -8.3773022, -0.9420204, 1.0084046
4: -4.7617931, -3.3518314, -4.8704977, -3.2272151, -1.3269677, 1.3649342
5: -8.3672180, -6.7967014, -8.4105492, -6.7741079, -1.1436386, 1.0993842
6: -5.9769497, -3.9463658, -6.0337806, -3.9245861, -1.6145687, 1.6182799
7: -4.2019429, -2.8157887, -4.2232685, -2.7517035, -1.3746932, 1.3405712
8: -3.7342639, -2.3025627, -3.7904797, -2.2832317, -1.1308688, 1.1078572
9: -11.0465937, -9.1663923, -11.2126026, -9.1222401, -1.4287829, 1.3680220

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6610722, upper bound: 0.6728251
time: 3.62 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6639518, upper bound: 0.6732133
time: 3.73 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 22.18 seconds
IS_B2_B2_A1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 22.18
Output dim: 1, lower bound: -0.6554208, upper bound: 0.6728252
IS_B2_B2_A1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 22.18
Output dim: 1, lower bound: -0.6582876, upper bound: 0.6732135
IS_B2_B2_A1_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 22.18
Output dim: 1, lower bound: -0.6610722, upper bound: 0.6728251
IS_B2_B2_A1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 22.18
Output dim: 1, lower bound: -0.6639518, upper bound: 0.6732133

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -8.7963305, -6.8454838, -8.9063845, -6.6767721, -1.6030412, 1.6002977
1: 2.0284414, 3.6395884, 1.9282928, 3.7280312, -1.2795944, 1.3222847
2: -5.3606272, -3.8578575, -5.4847841, -3.7642205, -1.1958289, 1.2157590
3: -10.1241684, -8.4177017, -10.1967373, -8.3780651, -0.9211152, 0.9765216
4: -4.6528325, -3.3540852, -4.8630295, -3.2274294, -1.2779450, 1.2742250
5: -8.3604364, -6.8891673, -8.4100466, -6.7802415, -1.0559409, 1.0788169
6: -5.9684391, -3.9792721, -6.0328155, -3.9273710, -1.5828514, 1.5987344
7: -4.1930618, -2.8370776, -4.2223830, -2.7533829, -1.3217525, 1.3161445
8: -3.7302113, -2.3509674, -3.7896314, -2.2865372, -1.0931382, 1.0740061
9: -11.0424976, -9.2389069, -11.2121773, -9.1278467, -1.3767483, 1.3364449

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 849

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6461527, upper bound: 0.6663336
time: 3.60 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6554051, upper bound: 0.6728096
time: 3.64 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -8.7996635, -6.8431053, -8.9066572, -6.6762867, -1.6055470, 1.6030192
1: 2.0247865, 3.6436334, 1.9276190, 3.7282810, -1.2835743, 1.3250489
2: -5.3653164, -3.8498659, -5.4859929, -3.7640307, -1.2002144, 1.2186348
3: -10.1271191, -8.4146490, -10.1969719, -8.3774738, -0.9234822, 0.9802489
4: -4.6550412, -3.3519154, -4.8633890, -3.2272739, -1.2808166, 1.2759016
5: -8.3622923, -6.8878784, -8.4102192, -6.7800117, -1.0583262, 1.0807621
6: -5.9751735, -3.9759862, -6.0331821, -3.9264820, -1.5868936, 1.6025467
7: -4.1956859, -2.8341522, -4.2228465, -2.7530987, -1.3258114, 1.3157890
8: -3.7342286, -2.3495970, -3.7903490, -2.2862606, -1.0974338, 1.0767338
9: -11.0482531, -9.2365246, -11.2124186, -9.1271734, -1.3798909, 1.3392787

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 849

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6490697, upper bound: 0.6667456
time: 3.68 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6582719, upper bound: 0.6731979
time: 3.77 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -8.7991858, -6.7485323, -8.9065094, -6.6708097, -1.6844888, 1.6244462
1: 1.9729714, 3.6404114, 1.9247208, 3.7280617, -1.3357172, 1.3509430
2: -5.4524326, -3.8540778, -5.4908314, -3.7640345, -1.2451408, 1.2881696
3: -10.1599598, -8.4146614, -10.1989393, -8.3779020, -0.9399247, 1.0058912
4: -4.7608299, -3.3522587, -4.8701358, -3.2273705, -1.3252206, 1.3633194
5: -8.3667326, -6.7973032, -8.4103765, -6.7743411, -1.1420579, 1.0979291
6: -5.9759736, -3.9487283, -6.0334105, -3.9254866, -1.6124096, 1.6156220
7: -4.2007113, -2.8165445, -4.2228012, -2.7519901, -1.3719132, 1.3380587
8: -3.7322903, -2.3032932, -3.7897530, -2.2835107, -1.1277037, 1.1056726
9: -11.0459452, -9.1681728, -11.2123575, -9.1229181, -1.4266062, 1.3652943

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 849

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6545433, upper bound: 0.6635698
time: 3.86 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6610565, upper bound: 0.6728096
time: 3.51 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.8025217, -6.7461724, -8.9067802, -6.6703210, -1.6869888, 1.6271563
1: 1.9693522, 3.6444702, 1.9240475, 3.7283115, -1.3396661, 1.3562083
2: -5.4570909, -3.8460879, -5.4920402, -3.7638450, -1.2494969, 1.2947717
3: -10.1628742, -8.4116173, -10.1991730, -8.3773079, -0.9422362, 1.0096036
4: -4.7630200, -3.3500779, -4.8704939, -3.2272158, -1.3280730, 1.3649921
5: -8.3686028, -6.7960196, -8.4105501, -6.7741103, -1.1444373, 1.0998695
6: -5.9827185, -3.9454410, -6.0337763, -3.9245973, -1.6164694, 1.6194439
7: -4.2033806, -2.8136592, -4.2232642, -2.7517061, -1.3759911, 1.3377161
8: -3.7363429, -2.3019342, -3.7904725, -2.2832336, -1.1303828, 1.1083813
9: -11.0517178, -9.1657982, -11.2126007, -9.1222439, -1.4305336, 1.3681228

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 849

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6574734, upper bound: 0.6639827
time: 3.81 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6639362, upper bound: 0.6731980
time: 3.85 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 28.85 seconds
IS_B2_B2_A1_B2_A2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6461527, upper bound: 0.6663336
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6554051, upper bound: 0.6728096
IS_B2_B2_A1_B2_A2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6490697, upper bound: 0.6667456
IS_B2_B2_A1_B2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6582719, upper bound: 0.6731979
IS_B2_B2_A1_B2_A2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6545433, upper bound: 0.6635698
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6610565, upper bound: 0.6728096
IS_B2_B2_A1_B2_A2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6574734, upper bound: 0.6639827
IS_B2_B2_A1_B2_A2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 28.85
Output dim: 1, lower bound: -0.6639362, upper bound: 0.6731980

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7963305, -6.8454838, -8.9063854, -6.6767812, -1.5780201, 1.5977068
1: 2.0284414, 3.6395884, 1.9285083, 3.7280297, -1.2839634, 1.3221390
2: -5.3606272, -3.8578575, -5.4845505, -3.7642262, -1.1952453, 1.2147102
3: -10.1241684, -8.4177017, -10.1967201, -8.3780670, -0.9208388, 0.9756862
4: -4.6528325, -3.3540852, -4.8627710, -3.2274296, -1.2777414, 1.2738166
5: -8.3604364, -6.8891673, -8.4100447, -6.7803807, -1.0550077, 1.0723414
6: -5.9684391, -3.9792721, -6.0327449, -3.9273701, -1.5807061, 1.5762966
7: -4.1930618, -2.8370776, -4.2223816, -2.7534795, -1.3208528, 1.3157330
8: -3.7302113, -2.3509674, -3.7896295, -2.2865944, -1.0930042, 1.0736434
9: -11.0424976, -9.2389069, -11.2121773, -9.1281328, -1.3763487, 1.3385899

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6554051, upper bound: 0.6718070
time: 3.72 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6554051, upper bound: 0.6728096
time: 3.67 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7996635, -6.8431053, -8.9066563, -6.6762943, -1.5805240, 1.6004291
1: 2.0247865, 3.6436334, 1.9278340, 3.7282810, -1.2879443, 1.3249028
2: -5.3653164, -3.8498659, -5.4857588, -3.7640362, -1.1996319, 1.2175887
3: -10.1271191, -8.4146490, -10.1969557, -8.3774738, -0.9232056, 0.9794165
4: -4.6550412, -3.3519154, -4.8631306, -3.2272754, -1.2806141, 1.2754939
5: -8.3622923, -6.8878784, -8.4102173, -6.7801495, -1.0573931, 1.0742897
6: -5.9751735, -3.9759862, -6.0331116, -3.9264822, -1.5847478, 1.5801105
7: -4.1956859, -2.8341522, -4.2228460, -2.7531948, -1.3249111, 1.3157885
8: -3.7342286, -2.3495970, -3.7903490, -2.2863169, -1.0972991, 1.0763714
9: -11.0482531, -9.2365246, -11.2124186, -9.1274624, -1.3794918, 1.3414249

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6578835, upper bound: 0.6703313
time: 3.73 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6578837, upper bound: 0.6703319
time: 3.92 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.7991867, -6.7485352, -8.9065094, -6.6708097, -1.6819081, 1.5994618
1: 1.9731035, 3.6404114, 1.9247208, 3.7280617, -1.3357162, 1.3556209
2: -5.4524307, -3.8540792, -5.4908314, -3.7640345, -1.2443099, 1.2881687
3: -10.1599588, -8.4146624, -10.1989393, -8.3779020, -0.9389558, 1.0056149
4: -4.7608280, -3.3522582, -4.8701358, -3.2273705, -1.3250051, 1.3631210
5: -8.3667307, -6.7973046, -8.4103765, -6.7743411, -1.1355312, 1.0970802
6: -5.9759698, -3.9487286, -6.0334105, -3.9254866, -1.5900688, 1.6135035
7: -4.2007113, -2.8165460, -4.2228012, -2.7519901, -1.3715060, 1.3372178
8: -3.7322893, -2.3032932, -3.7897530, -2.2835107, -1.1278669, 1.1056018
9: -11.0459442, -9.1681738, -11.2123575, -9.1229181, -1.4294040, 1.3651242

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6610565, upper bound: 0.6718069
time: 3.64 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6610565, upper bound: 0.6728096
time: 3.68 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.8025227, -6.7461739, -8.9067802, -6.6703210, -1.6844087, 1.6021719
1: 1.9694872, 3.6444712, 1.9240475, 3.7283115, -1.3396652, 1.3608859
2: -5.4570904, -3.8460889, -5.4920402, -3.7638450, -1.2486658, 1.2942061
3: -10.1628742, -8.4116173, -10.1991730, -8.3773079, -0.9412766, 1.0093279
4: -4.7630177, -3.3500776, -4.8704939, -3.2272158, -1.3278570, 1.3647931
5: -8.3686018, -6.7960205, -8.4105501, -6.7741103, -1.1379147, 1.0990205
6: -5.9827137, -3.9454410, -6.0337763, -3.9245973, -1.5941281, 1.6173263
7: -4.2033801, -2.8136601, -4.2232642, -2.7517061, -1.3755844, 1.3366368
8: -3.7363443, -2.3019342, -3.7904725, -2.2832336, -1.1305456, 1.1083105
9: -11.0517168, -9.1658001, -11.2126007, -9.1222439, -1.4333324, 1.3679525

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6635349, upper bound: 0.6703313
time: 3.74 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6448292, upper bound: 0.6638074
time: 4.26 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6639353, upper bound: 0.6731969
time: 3.96 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 44.24 seconds
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6554051, upper bound: 0.6718070
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6554051, upper bound: 0.6728096
IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6578835, upper bound: 0.6703313
IS_B2_B2_A1_B2_A2_A1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6578837, upper bound: 0.6703319
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6610565, upper bound: 0.6718069
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6610565, upper bound: 0.6728096
IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6448292, upper bound: 0.6638074
IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 44.24
Output dim: 1, lower bound: -0.6639353, upper bound: 0.6731969

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.7963305, -6.8454838, -8.9059353, -6.6775885, -1.5771799, 1.5970442
1: 2.0284414, 3.6395884, 1.9296260, 3.7276182, -1.2833095, 1.3209381
2: -5.3606272, -3.8578575, -5.4825535, -3.7645409, -1.1948462, 1.2128146
3: -10.1241684, -8.4177017, -10.1963310, -8.3790483, -0.9196389, 0.9748510
4: -4.6528325, -3.3540852, -4.8621759, -3.2276847, -1.2771256, 1.2730141
5: -8.3604364, -6.8891673, -8.4097586, -6.7807598, -1.0544147, 1.0716248
6: -5.9684391, -3.9792721, -6.0321379, -3.9288447, -1.5794258, 1.5755274
7: -4.1930618, -2.8370776, -4.2216139, -2.7539513, -1.3196332, 1.3146026
8: -3.7302113, -2.3509674, -3.7884393, -2.2870507, -1.0923986, 1.0718443
9: -11.0424976, -9.2389069, -11.2117796, -9.1292448, -1.3750551, 1.3378114

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6459716, upper bound: 0.6529110
time: 4.02 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6554042, upper bound: 0.6718060
time: 3.69 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.7963305, -6.8454838, -8.9092789, -6.6752844, -1.5794201, 1.5989749
1: 2.0284414, 3.6395884, 1.9260702, 3.7316613, -1.2852702, 1.3244603
2: -5.3606272, -3.8578575, -5.4871464, -3.7565391, -1.1963785, 1.2167804
3: -10.1241684, -8.4177017, -10.1992731, -8.3760424, -0.9227474, 0.9765284
4: -4.6528325, -3.3540852, -4.8643293, -3.2255116, -1.2782269, 1.2751954
5: -8.3604364, -6.8891673, -8.4116039, -6.7794900, -1.0557439, 1.0736240
6: -5.9684391, -3.9792721, -6.0389209, -3.9255710, -1.5826459, 1.5786526
7: -4.1930618, -2.8370776, -4.2243071, -2.7510357, -1.3217216, 1.3159676
8: -3.7302113, -2.3509674, -3.7924976, -2.2857056, -1.0938745, 1.0757191
9: -11.0424976, -9.2389069, -11.2175722, -9.1268568, -1.3772836, 1.3400385

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6459716, upper bound: 0.6538637
time: 3.94 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6554042, upper bound: 0.6728087
time: 3.81 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.7991867, -6.7485352, -8.9060612, -6.6716142, -1.6810718, 1.5987995
1: 1.9731035, 3.6404114, 1.9258366, 3.7276468, -1.3350611, 1.3543772
2: -5.4524307, -3.8540792, -5.4888358, -3.7643495, -1.2439103, 1.2861688
3: -10.1599588, -8.4146624, -10.1985493, -8.3788815, -0.9377558, 1.0048119
4: -4.7608280, -3.3522582, -4.8695397, -3.2276268, -1.3243876, 1.3623247
5: -8.3667307, -6.7973046, -8.4100904, -6.7747202, -1.1349404, 1.0963682
6: -5.9759698, -3.9487286, -6.0328021, -3.9269593, -1.5887866, 1.6127429
7: -4.2007113, -2.8165460, -4.2220316, -2.7524614, -1.3703055, 1.3360882
8: -3.7322893, -2.3032932, -3.7885637, -2.2839665, -1.1276382, 1.1038010
9: -11.0459442, -9.1681738, -11.2119579, -9.1240273, -1.4280291, 1.3643453

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6420579, upper bound: 0.6623089
time: 4.11 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6610556, upper bound: 0.6718060
time: 3.67 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7991867, -6.7485352, -8.9094057, -6.6693134, -1.6832867, 1.6007304
1: 1.9731035, 3.6404114, 1.9222822, 3.7316923, -1.3370223, 1.3581735
2: -5.4524307, -3.8540792, -5.4934263, -3.7563467, -1.2454424, 1.2908788
3: -10.1599588, -8.4146624, -10.2014923, -8.3758783, -0.9408643, 1.0064216
4: -4.7608280, -3.3522582, -4.8716908, -3.2254515, -1.3254900, 1.3644514
5: -8.3667307, -6.7973046, -8.4119339, -6.7734518, -1.1362617, 1.0983638
6: -5.9759698, -3.9487286, -6.0395870, -3.9236846, -1.5919962, 1.6158700
7: -4.2007113, -2.8165460, -4.2247281, -2.7495482, -1.3723340, 1.3368368
8: -3.7322893, -2.3032932, -3.7926207, -2.2826266, -1.1274116, 1.1076772
9: -11.0459442, -9.1681738, -11.2177525, -9.1216393, -1.4308167, 1.3665736

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6420579, upper bound: 0.6633739
time: 4.01 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6610556, upper bound: 0.6728087
time: 3.58 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8025217, -6.7461729, -8.9067812, -6.6703215, -1.6843863, 1.6021516
1: 1.9694896, 3.6444702, 1.9240556, 3.7283106, -1.3396554, 1.3609096
2: -5.4570875, -3.8460894, -5.4920354, -3.7638450, -1.2486639, 1.2941992
3: -10.1628742, -8.4116173, -10.1991749, -8.3773088, -0.9412608, 1.0093197
4: -4.7630172, -3.3500772, -4.8704934, -3.2272160, -1.3278458, 1.3647900
5: -8.3686018, -6.7960224, -8.4105492, -6.7741179, -1.1378973, 1.0990129
6: -5.9827037, -3.9454408, -6.0337539, -3.9245965, -1.5941057, 1.6172669
7: -4.2033792, -2.8136640, -4.2232637, -2.7517142, -1.3755636, 1.3366315
8: -3.7363424, -2.3019342, -3.7904720, -2.2832355, -1.1305172, 1.1083019
9: -11.0517159, -9.1658020, -11.2125978, -9.1222515, -1.4333410, 1.3679469

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 916

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6635340, upper bound: 0.6703303
time: 3.61 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6635341, upper bound: 0.6537557
time: 7.29 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 32.02 seconds
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6459716, upper bound: 0.6529110
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6554042, upper bound: 0.6718060
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6459716, upper bound: 0.6538637
IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6554042, upper bound: 0.6728087
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6420579, upper bound: 0.6623089
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6610556, upper bound: 0.6718060
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6420579, upper bound: 0.6633739
IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6610556, upper bound: 0.6728087
IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6635340, upper bound: 0.6703303
IS_B2_B2_A1_B2_A2_A1_A2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 11, time: 32.02
Output dim: 1, lower bound: -0.6635341, upper bound: 0.6537557

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.7963314, -6.8454847, -8.9059334, -6.6775875, -1.5771599, 1.5970199
1: 2.0284510, 3.6395884, 1.9296293, 3.7276173, -1.2832949, 1.3209274
2: -5.3606215, -3.8578572, -5.4825521, -3.7645411, -1.1948392, 1.2128110
3: -10.1241684, -8.4176989, -10.1963310, -8.3790483, -0.9196303, 0.9748356
4: -4.6528325, -3.3540850, -4.8621755, -3.2276850, -1.2771220, 1.2730031
5: -8.3604345, -6.8891730, -8.4097595, -6.7807612, -1.0544078, 1.0716068
6: -5.9684172, -3.9792717, -6.0321255, -3.9288447, -1.5793657, 1.5755043
7: -4.1930618, -2.8370862, -4.2216129, -2.7539561, -1.3196239, 1.3145826
8: -3.7302122, -2.3509688, -3.7884398, -2.2870498, -1.0923905, 1.0718353
9: -11.0424967, -9.2389126, -11.2117777, -9.1292496, -1.3750486, 1.3377942

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6546118, upper bound: 0.6714087
time: 3.96 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6546118, upper bound: 0.6703299
time: 4.09 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.7963314, -6.8454847, -8.9092789, -6.6752853, -1.5794001, 1.5989509
1: 2.0284510, 3.6395884, 1.9260726, 3.7316618, -1.2852559, 1.3244500
2: -5.3606215, -3.8578572, -5.4871445, -3.7565384, -1.1963718, 1.2167773
3: -10.1241684, -8.4176989, -10.1992741, -8.3760433, -0.9227393, 0.9765129
4: -4.6528325, -3.3540850, -4.8643279, -3.2255111, -1.2782238, 1.2751832
5: -8.3604345, -6.8891730, -8.4116030, -6.7794933, -1.0557373, 1.0736060
6: -5.9684172, -3.9792717, -6.0389090, -3.9255724, -1.5825868, 1.5786302
7: -4.1930618, -2.8370862, -4.2243071, -2.7510397, -1.3217125, 1.3159888
8: -3.7302122, -2.3509688, -3.7924976, -2.2857075, -1.0938663, 1.0757103
9: -11.0424967, -9.2389126, -11.2175722, -9.1268587, -1.3772779, 1.3400218

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6141
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6364517, upper bound: 0.6633739
time: 4.19 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6364517, upper bound: 0.6728066
time: 5.67 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.7991858, -6.7485347, -8.9060602, -6.6716175, -1.6810489, 1.5987799
1: 1.9731083, 3.6404128, 1.9258451, 3.7276468, -1.3350506, 1.3544006
2: -5.4524288, -3.8540785, -5.4888315, -3.7643499, -1.2439070, 1.2861725
3: -10.1599569, -8.4146624, -10.1985502, -8.3788815, -0.9377408, 1.0048035
4: -4.7608280, -3.3522587, -4.8695383, -3.2276263, -1.3243766, 1.3623219
5: -8.3667316, -6.7973070, -8.4100895, -6.7747273, -1.1349237, 1.0963610
6: -5.9759588, -3.9487295, -6.0327787, -3.9269593, -1.5887632, 1.6126840
7: -4.2007103, -2.8165493, -4.2220325, -2.7524693, -1.3702860, 1.3360791
8: -3.7322893, -2.3032947, -3.7885604, -2.2839680, -1.1276095, 1.1037934
9: -11.0459433, -9.1681757, -11.2119579, -9.1240358, -1.4280374, 1.3643395

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 6141
type: B, layer: 1, pos: 442

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6602427, upper bound: 0.6714090
time: 4.06 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6602427, upper bound: 0.6703298
time: 4.26 seconds

## BFS IS instance: IS_B2_B2_A1_B2_A2_A1_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.7991858, -6.7485347, -8.9094057, -6.6693144, -1.6832647, 1.6007106
1: 1.9731083, 3.6404128, 1.9222903, 3.7316923, -1.3370128, 1.3581966
2: -5.4524288, -3.8540785, -5.4934220, -3.7563477, -1.2454395, 1.2908831
3: -10.1599569, -8.4146624, -10.2014923, -8.3758783, -0.9408491, 1.0064132
4: -4.7608280, -3.3522587, -4.8716908, -3.2254522, -1.3254786, 1.3644483
5: -8.3667316, -6.7973070, -8.4119349, -6.7734575, -1.1362445, 1.0983571
6: -5.9759588, -3.9487295, -6.0395641, -3.9236856, -1.5919733, 1.6158113
7: -4.2007103, -2.8165493, -4.2247272, -2.7495553, -1.3723137, 1.3368325
8: -3.7322893, -2.3032947, -3.7926207, -2.2826257, -1.1273830, 1.1076696
9: -11.0459433, -9.1681757, -11.2177534, -9.1216469, -1.4308255, 1.3665679

Time for backsubstitution: 14.66 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2421.97 seconds
