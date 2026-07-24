## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.20377202038
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.8367071, 3.8367071)
1: (-13.2111492, -8.7825651, -13.2111492, -8.7825651, -4.4285841, 4.4285841)
2: (-8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100)
3: (-9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.6361666, 4.6361666)
4: (-11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687)
5: (-0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4574890, 3.4574890)
6: (4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096)
7: (-18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.7471981, 3.7471981)
8: (0.0874861, 4.0993404, 0.0874861, 4.0993404, -4.0118542, 4.0118542)
9: (-8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.1832142, 3.1832142)

## BASE Result
execution time: IAR + LP analysis = 14.96 + 32.28 = 47.25 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.75 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.050609588623047
rel_dist={6: [-1.840513682283781, 1.840511023435588]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.050609588623047
rel_dist={6: [-1.4762376626074394, 1.476239686090869]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.956814765930176
rel_dist={6: [-1.2041204236983045, 1.204119556210955]}

## Binary Search Result
Binary search time: 152.26 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 3400.50 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404950, upper bound: 1.8351101
time: 6.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404950, upper bound: 1.8404944
time: 5.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.64
Output dim: 6, lower bound: -1.8404950, upper bound: 1.8351101
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.64
Output dim: 6, lower bound: -1.8404950, upper bound: 1.8404944

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105403, -9.5425024, -5.7090302, -3.5080605, 3.6463132
1: -13.1846399, -8.7899218, -13.2049541, -8.7842789, -4.0118570, 4.0443740
2: -8.0927343, -4.3504028, -8.1217937, -4.3396597, -3.7530746, 3.7713909
3: -9.7975979, -5.1735291, -9.8004036, -5.1670346, -4.4802790, 4.4761395
4: -11.0601044, -7.0887394, -11.0672636, -7.0809603, -3.9791441, 3.9785242
5: -0.2551122, 3.1854463, -0.2607970, 3.1927176, -3.4187894, 3.4257898
6: 4.4715471, 7.5042758, 4.4659367, 7.5123577, -3.0408106, 3.0383391
7: -18.0248909, -14.3027716, -18.0373383, -14.2959843, -3.4903555, 3.5076289
8: 0.0937277, 4.0807061, 0.0889307, 4.0949650, -3.9816418, 3.9086123
9: -8.8935957, -5.7405071, -8.8994093, -5.7232976, -3.0373549, 2.9991028

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351074, upper bound: 1.8351073
time: 5.07 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351074, upper bound: 1.8351072
time: 4.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961126, -9.5452633, -5.7085772, -3.6938753, 3.6659150
1: -13.2471294, -8.7146473, -13.2111111, -8.7825737, -4.0934815, 4.0883107
2: -8.1452150, -4.2182059, -8.1305704, -4.3364363, -3.8087788, 3.9123645
3: -9.8206444, -5.0881243, -9.8012657, -5.1651144, -4.5119305, 4.6018815
4: -11.1171827, -7.0695877, -11.0695467, -7.0786047, -4.0385780, 3.9999590
5: -0.3109841, 3.2038767, -0.2625225, 3.1949501, -3.4912939, 3.4371705
6: 4.4145656, 7.5255613, 4.4642410, 7.5148315, -3.1002660, 3.0613203
7: -18.0553322, -14.2519770, -18.0411186, -14.2939520, -3.5219898, 3.5566421
8: 0.0272651, 4.1195259, 0.0874935, 4.0993271, -4.0478735, 4.0121355
9: -8.9681721, -5.7078457, -8.9012594, -5.7180867, -3.0810509, 3.0319648

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404930, upper bound: 1.8384943
time: 4.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404931, upper bound: 1.8404924
time: 5.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 6, lower bound: -1.8351074, upper bound: 1.8351073
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 6, lower bound: -1.8351074, upper bound: 1.8351072
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 6, lower bound: -1.8404930, upper bound: 1.8384943
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 6, lower bound: -1.8404931, upper bound: 1.8404924

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105403, -9.5346851, -5.7105403, -3.5046883, 3.5046883
1: -13.1846399, -8.7899218, -13.1846399, -8.7899218, -4.0205097, 4.0205092
2: -8.0927343, -4.3504028, -8.0927343, -4.3504028, -3.7423315, 3.7423315
3: -9.7975979, -5.1735291, -9.7975979, -5.1735291, -4.4671030, 4.4671025
4: -11.0601044, -7.0887394, -11.0601044, -7.0887394, -3.9713650, 3.9713650
5: -0.2551122, 3.1854463, -0.2551122, 3.1854463, -3.4200182, 3.4200182
6: 4.4715471, 7.5042758, 4.4715471, 7.5042758, -3.0327287, 3.0327287
7: -18.0248909, -14.3027716, -18.0248909, -14.3027716, -3.4946775, 3.4946771
8: 0.0937277, 4.0807061, 0.0937277, 4.0807061, -3.9048100, 3.9048095
9: -8.8935957, -5.7405071, -8.8935957, -5.7405071, -3.0205626, 3.0205624

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8331099
time: 4.69 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8351084
time: 4.96 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105403, -9.5826025, -5.6961126, -3.5200939, 3.6861014
1: -13.1846399, -8.7899218, -13.2471294, -8.7146473, -4.0568743, 4.0845933
2: -8.0927343, -4.3504028, -8.1452150, -4.2182059, -3.8745284, 3.7948122
3: -9.7975979, -5.1735291, -9.8206444, -5.0881243, -4.5654736, 4.4946899
4: -11.0601044, -7.0887394, -11.1171827, -7.0695877, -3.9905167, 4.0284433
5: -0.2551122, 3.1854463, -0.3109841, 3.2038767, -3.4308891, 3.4907150
6: 4.4715471, 7.5042758, 4.4145656, 7.5255613, -3.0540142, 3.0897102
7: -18.0248909, -14.3027716, -18.0553322, -14.2519770, -3.5372000, 3.5246005
8: 0.0937277, 4.0807061, 0.0272651, 4.1195259, -4.0058260, 3.9609475
9: -8.8935957, -5.7405071, -8.9681721, -5.7078457, -3.0514035, 3.0583489

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5860

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8331070, upper bound: 1.8351081
time: 5.05 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351054, upper bound: 1.8351082
time: 4.95 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.5810862, -5.6971622, -9.5288372, -5.7166462, -3.6808071, 3.6460085
1: -13.2450981, -8.7211409, -13.1868401, -8.8050957, -4.0693245, 4.0525331
2: -8.1438169, -4.2204733, -8.1166258, -4.3494244, -3.7943926, 3.8961525
3: -9.8193817, -5.0897155, -9.7856121, -5.1766047, -4.4966879, 4.5820842
4: -11.1140604, -7.0735970, -11.0459728, -7.0996704, -4.0143900, 3.9723759
5: -0.3077459, 3.2029142, -0.2461510, 3.1862841, -3.4790154, 3.4194655
6: 4.4198179, 7.5251837, 4.4851904, 7.5064173, -3.0865993, 3.0399933
7: -18.0536118, -14.2619219, -18.0176048, -14.3323174, -3.4808846, 3.5228481
8: 0.0292166, 4.1181798, 0.1031935, 4.0875969, -4.0314660, 3.9889464
9: -8.9662170, -5.7122583, -8.8883629, -5.7373600, -3.0568810, 3.0128641

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351074, upper bound: 1.8384945
time: 4.72 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8384946
time: 4.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961136, -9.5452623, -5.7085772, -3.6986074, 3.6601543
1: -13.2471266, -8.7146492, -13.2111082, -8.7825813, -4.0805206, 4.0790091
2: -8.1452131, -4.2182040, -8.1305714, -4.3364387, -3.8087745, 3.9123673
3: -9.8206463, -5.0881248, -9.8012638, -5.1651134, -4.5108280, 4.6011581
4: -11.1171837, -7.0695877, -11.0695438, -7.0786085, -4.0385752, 3.9999561
5: -0.3109806, 3.2038770, -0.2625196, 3.1949494, -3.4918165, 3.4370914
6: 4.4145675, 7.5255613, 4.4642472, 7.5148296, -3.1002622, 3.0613141
7: -18.0553322, -14.2519779, -18.0411186, -14.2939625, -3.5129032, 3.5526347
8: 0.0272665, 4.1195240, 0.0874939, 4.0993280, -4.0404091, 4.0131230
9: -8.9681702, -5.7078457, -8.9012594, -5.7180901, -3.0760946, 3.0319622

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8404923
time: 5.22 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8404928
time: 4.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.82 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8331099
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8351084
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8331070, upper bound: 1.8351081
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351054, upper bound: 1.8351082
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351074, upper bound: 1.8384945
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8384946
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8404923
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.82
Output dim: 6, lower bound: -1.8351055, upper bound: 1.8404928

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.5331736, -5.7115870, -9.5190468, -5.7185955, -3.4934626, 3.4878254
1: -13.1826153, -8.7964096, -13.1603251, -8.8122578, -3.9974785, 3.9872179
2: -8.0913391, -4.3526754, -8.0787745, -4.3632579, -3.7280812, 3.7260990
3: -9.7963390, -5.1751170, -9.7820282, -5.1850348, -4.4517908, 4.4472833
4: -11.0569687, -7.0927496, -11.0377274, -7.1098089, -3.9471598, 3.9449778
5: -0.2518771, 3.1844783, -0.2387795, 3.1769717, -3.4078665, 3.4024868
6: 4.4767523, 7.5038967, 4.4925060, 7.4958830, -3.0191307, 3.0113907
7: -18.0231552, -14.3127193, -18.0013733, -14.3409834, -3.4535208, 3.4606051
8: 0.0956668, 4.0793686, 0.1094306, 4.0692348, -3.8900023, 3.8813696
9: -8.8916626, -5.7449121, -8.8810453, -5.7597656, -2.9962864, 3.0021365

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350956, upper bound: 1.8287835
time: 4.89 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350952, upper bound: 1.8330961
time: 4.89 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105403, -9.5346842, -5.7105408, -3.5055342, 3.5029383
1: -13.1846399, -8.7899246, -13.1846390, -8.7899284, -4.0065584, 4.0205069
2: -8.0927353, -4.3504019, -8.0927334, -4.3504043, -3.7423310, 3.7423315
3: -9.7975969, -5.1735306, -9.7975969, -5.1735311, -4.4658737, 4.4665790
4: -11.0601025, -7.0887403, -11.0601025, -7.0887427, -3.9713597, 3.9713621
5: -0.2551105, 3.1854448, -0.2551088, 3.1854455, -3.4211445, 3.4198546
6: 4.4715481, 7.5042753, 4.4715533, 7.5042758, -3.0327277, 3.0327220
7: -18.0248909, -14.3027763, -18.0248890, -14.3027792, -3.4848347, 3.4946761
8: 0.0937277, 4.0807066, 0.0937289, 4.0807047, -3.9031324, 3.9045677
9: -8.8935957, -5.7405081, -8.8935966, -5.7405100, -3.0192642, 3.0205600

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8350953
time: 4.88 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350948, upper bound: 1.8350950
time: 4.64 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5190468, -5.7185955, -9.5810862, -5.6971622, -3.5032368, 3.6730289
1: -13.1603251, -8.8122578, -13.2450981, -8.7211409, -4.0211287, 4.0614543
2: -8.0787745, -4.3632579, -8.1438169, -4.2204733, -3.8583012, 3.7805591
3: -9.7820282, -5.1850348, -9.8193817, -5.0897155, -4.5456505, 4.4794674
4: -11.0377274, -7.1098089, -11.1140604, -7.0735970, -3.9641304, 4.0042515
5: -0.2387795, 3.1769717, -0.3077459, 3.2029142, -3.4132299, 3.4785714
6: 4.4925060, 7.4958830, 4.4198179, 7.5251837, -3.0326777, 3.0760651
7: -18.0013733, -14.3409834, -18.0536118, -14.2619219, -3.5034018, 3.4834571
8: 0.1094306, 4.0692348, 0.0292166, 4.1181798, -3.9826269, 3.9457412
9: -8.8810453, -5.7597656, -8.9662170, -5.7122583, -3.0329666, 3.0342586

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8341735, upper bound: 1.8350937
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8384792, upper bound: 1.8350930
time: 5.84 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5346842, -5.7105408, -9.5826025, -5.6961136, -3.5183434, 3.6908360
1: -13.1846390, -8.7899284, -13.2471266, -8.7146492, -4.0475702, 4.0706415
2: -8.0927334, -4.3504043, -8.1452131, -4.2182040, -3.8745294, 3.7948089
3: -9.7975969, -5.1735311, -9.8206463, -5.0881248, -4.5648737, 4.4936171
4: -11.0601025, -7.0887427, -11.1171837, -7.0695877, -3.9905148, 4.0284410
5: -0.2551088, 3.1854455, -0.3109806, 3.2038770, -3.4308143, 3.4918432
6: 4.4715533, 7.5042758, 4.4145675, 7.5255613, -3.0540080, 3.0897083
7: -18.0248890, -14.3027792, -18.0553322, -14.2519779, -3.5352187, 3.5147567
8: 0.0937289, 4.0807047, 0.0272665, 4.1195240, -4.0068130, 3.9548883
9: -8.8935966, -5.7405100, -8.9681702, -5.7078457, -3.0514011, 3.0533934

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404780, upper bound: 1.8307799
time: 5.09 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404776, upper bound: 1.8350928
time: 4.96 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.5810862, -5.6971622, -9.5190468, -5.7185955, -3.6730289, 3.5032368
1: -13.2450981, -8.7211409, -13.1603251, -8.8122578, -4.0614548, 4.0211287
2: -8.1438169, -4.2204733, -8.0787745, -4.3632579, -3.7805591, 3.8583012
3: -9.8193817, -5.0897155, -9.7820282, -5.1850348, -4.4794674, 4.5456505
4: -11.1140604, -7.0735970, -11.0377274, -7.1098089, -4.0042515, 3.9641304
5: -0.3077459, 3.2029142, -0.2387795, 3.1769717, -3.4785719, 3.4132299
6: 4.4198179, 7.5251837, 4.4925060, 7.4958830, -3.0760651, 3.0326777
7: -18.0536118, -14.2619219, -18.0013733, -14.3409834, -3.4834566, 3.5034018
8: 0.0292166, 4.1181798, 0.1094306, 4.0692348, -3.9457412, 3.9826264
9: -8.9662170, -5.7122583, -8.8810453, -5.7597656, -3.0342588, 3.0329664

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350910, upper bound: 1.8341735
time: 4.82 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350906, upper bound: 1.8384790
time: 5.00 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.5810862, -5.6971622, -9.5661783, -5.7041712, -3.6907835, 3.6839042
1: -13.2450981, -8.7211409, -13.2228756, -8.7371750, -4.0933456, 4.0840707
2: -8.1438169, -4.2204733, -8.1312160, -4.2312269, -3.9125900, 3.9107428
3: -9.8193817, -5.0897155, -9.8049612, -5.0997496, -4.6004810, 4.5959206
4: -11.1140604, -7.0735970, -11.0935974, -7.0906558, -4.0234046, 4.0200005
5: -0.3077459, 3.2029142, -0.2946267, 3.1952267, -3.4848928, 3.4795542
6: 4.4198179, 7.5251837, 4.4356613, 7.5171566, -3.0973387, 3.0895224
7: -18.0536118, -14.2619219, -18.0318909, -14.2903366, -3.5301409, 3.5374942
8: 0.0292166, 4.1181798, 0.0430305, 4.1077852, -4.0338688, 4.0265255
9: -8.9662170, -5.7122583, -8.9552460, -5.7271318, -3.0456614, 3.0507562

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350910, upper bound: 1.8341736
time: 4.95 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350906, upper bound: 1.8384794
time: 4.94 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961136, -9.5346842, -5.7105408, -3.6908360, 3.5183434
1: -13.2471266, -8.7146492, -13.1846390, -8.7899284, -4.0706415, 4.0475702
2: -8.1452131, -4.2182040, -8.0927334, -4.3504043, -3.7948089, 3.8745294
3: -9.8206463, -5.0881248, -9.7975969, -5.1735311, -4.4936171, 4.5648742
4: -11.1171837, -7.0695877, -11.0601025, -7.0887427, -4.0284410, 3.9905148
5: -0.3109806, 3.2038770, -0.2551088, 3.1854455, -3.4918432, 3.4308143
6: 4.4145675, 7.5255613, 4.4715533, 7.5042758, -3.0897083, 3.0540080
7: -18.0553322, -14.2519779, -18.0248890, -14.3027792, -3.5147562, 3.5352182
8: 0.0272665, 4.1195240, 0.0937289, 4.0807047, -3.9548893, 4.0068130
9: -8.9681702, -5.7078457, -8.8935966, -5.7405100, -3.0533934, 3.0514011

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8404777
time: 4.61 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350902, upper bound: 1.8404774
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961136, -9.5826015, -5.6961126, -3.7085538, 3.6980734
1: -13.2471266, -8.7146492, -13.2471256, -8.7146549, -4.1045580, 4.1135864
2: -8.1452131, -4.2182040, -8.1452131, -4.2182064, -3.9270067, 3.9270091
3: -9.8206463, -5.0881248, -9.8206444, -5.0881262, -4.6147146, 4.6150060
4: -11.1171837, -7.0695877, -11.1171818, -7.0695896, -4.0475941, 4.0475941
5: -0.3109806, 3.2038770, -0.3109784, 3.2038751, -3.4976883, 3.4970875
6: 4.4145675, 7.5255613, 4.4145718, 7.5255613, -3.1109939, 3.1109896
7: -18.0553322, -14.2519779, -18.0553322, -14.2519817, -3.5621586, 3.5672693
8: 0.0272665, 4.1195240, 0.0272675, 4.1195230, -4.0468607, 4.0508008
9: -8.9681702, -5.7078457, -8.9681683, -5.7078476, -3.0651908, 3.0697975

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8404782
time: 4.96 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350902, upper bound: 1.8404778
time: 4.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.06 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350956, upper bound: 1.8287835
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350952, upper bound: 1.8330961
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8350953
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350948, upper bound: 1.8350950
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8341735, upper bound: 1.8350937
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8384792, upper bound: 1.8350930
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8404780, upper bound: 1.8307799
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8404776, upper bound: 1.8350928
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350910, upper bound: 1.8341735
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350906, upper bound: 1.8384790
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350910, upper bound: 1.8341736
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350906, upper bound: 1.8384794
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8404777
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350902, upper bound: 1.8404774
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8404782
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.06
Output dim: 6, lower bound: -1.8350902, upper bound: 1.8404778

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.5331736, -5.7115870, -9.5106325, -5.7205982, -3.4903779, 3.4809947
1: -13.1826153, -8.7964096, -13.1549025, -8.8600349, -3.9503417, 3.9808626
2: -8.0913391, -4.3526754, -8.0685768, -4.3693171, -3.7220221, 3.7159014
3: -9.7963390, -5.1751170, -9.7514420, -5.1897492, -4.4442654, 4.4149175
4: -11.0569687, -7.0927496, -11.0311842, -7.1375422, -3.9194264, 3.9384346
5: -0.2518771, 3.1844783, -0.1968565, 3.1748157, -3.4055214, 3.3601742
6: 4.4767523, 7.5038967, 4.5073214, 7.4933276, -3.0165753, 2.9965754
7: -18.0231552, -14.3127193, -17.9988384, -14.3511667, -3.4399605, 3.4586654
8: 0.0956668, 4.0793686, 0.1162175, 4.0473661, -3.8612366, 3.8673439
9: -8.8916626, -5.7449121, -8.8550072, -5.7684307, -2.9864531, 2.9731798

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8287834
time: 4.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8287837
time: 4.77 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5331669, -5.7115879, -9.5384808, -5.6820860, -3.5325899, 3.5137248
1: -13.1826105, -8.7964306, -13.3064976, -8.7983084, -4.0125895, 4.0593596
2: -8.0913324, -4.3526821, -8.1014614, -4.3408709, -3.7504616, 3.7487793
3: -9.7963266, -5.1751213, -9.7938185, -5.0867329, -4.5439129, 4.4580274
4: -11.0569630, -7.0927730, -11.1180458, -7.0958443, -3.9611187, 4.0252728
5: -0.2518487, 3.1844778, -0.2681775, 3.2897587, -3.4696202, 3.4345560
6: 4.4767666, 7.5038977, 4.4577813, 7.5338411, -3.0570745, 3.0461164
7: -18.0231552, -14.3127327, -18.0327682, -14.3295116, -3.4643707, 3.4929209
8: 0.0956720, 4.0793562, 0.0280328, 4.0848241, -3.8998337, 3.9492517
9: -8.8916473, -5.7449164, -8.8882856, -5.6778417, -3.0627728, 3.0075796

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278850, upper bound: 1.8330882
time: 4.57 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350871, upper bound: 1.8330883
time: 4.99 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5262566, -5.7125421, -9.5346842, -5.7105408, -3.4986906, 3.4998574
1: -13.1792192, -8.8376961, -13.1846390, -8.7899284, -4.0002069, 3.9733047
2: -8.0825396, -4.3564639, -8.0927334, -4.3504043, -3.7321353, 3.7362695
3: -9.7670126, -5.1782527, -9.7975969, -5.1735311, -4.4335051, 4.4590487
4: -11.0535545, -7.1164651, -11.0601025, -7.0887427, -3.9648118, 3.9436374
5: -0.2131996, 3.1832910, -0.2551088, 3.1854455, -3.3788500, 3.4175119
6: 4.4864011, 7.5017223, 4.4715533, 7.5042758, -3.0178747, 3.0301690
7: -18.0223560, -14.3129520, -18.0248890, -14.3027792, -3.4828930, 3.4811339
8: 0.1005214, 4.0588446, 0.0937289, 4.0807047, -3.8890839, 3.8758001
9: -8.8675709, -5.7491770, -8.8935966, -5.7405100, -2.9903183, 3.0107229

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8307818
time: 5.03 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8350949
time: 4.93 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5541363, -5.6740274, -9.5346775, -5.7105412, -3.5312824, 3.5420871
1: -13.3308325, -8.7760048, -13.1846352, -8.7899475, -4.0805006, 4.0355387
2: -8.1154575, -4.3280349, -8.0927267, -4.3504081, -3.7650495, 3.7646918
3: -9.8094025, -5.0752344, -9.7975845, -5.1735344, -4.4765234, 4.5537672
4: -11.1404095, -7.0748119, -11.0600967, -7.0887675, -4.0516419, 3.9852848
5: -0.2844250, 3.2982771, -0.2550809, 3.1854451, -3.4531336, 3.4806950
6: 4.4368773, 7.5422440, 4.4715681, 7.5042748, -3.0673976, 3.0706758
7: -18.0562763, -14.2913532, -18.0248871, -14.3027897, -3.5171585, 3.5056067
8: 0.0123024, 4.0963411, 0.0937343, 4.0806932, -3.9674754, 3.9143543
9: -8.9008398, -5.6585288, -8.8935823, -5.7405148, -3.0247116, 3.0835788

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350871, upper bound: 1.8278848
time: 4.66 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350870, upper bound: 1.8350868
time: 4.69 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -9.5106325, -5.7205982, -9.5810862, -5.6971622, -3.4964056, 3.6697693
1: -13.1549025, -8.8600349, -13.2450981, -8.7211409, -4.0151138, 4.0143185
2: -8.0685768, -4.3693171, -8.1438169, -4.2204733, -3.8481035, 3.7744999
3: -9.7514420, -5.1897492, -9.8193817, -5.0897155, -4.5132847, 4.4721804
4: -11.0311842, -7.1375422, -11.1140604, -7.0735970, -3.9575872, 3.9765182
5: -0.1968565, 3.1748157, -0.3077459, 3.2029142, -3.3710632, 3.4762259
6: 4.5073214, 7.4933276, 4.4198179, 7.5251837, -3.0178623, 3.0735097
7: -17.9988384, -14.3511667, -18.0536118, -14.2619219, -3.5014591, 3.4698963
8: 0.1162175, 4.0473661, 0.0292166, 4.1181798, -3.9692841, 3.9169579
9: -8.8550072, -5.7684307, -8.9662170, -5.7122583, -3.0040102, 3.0250080

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8341735, upper bound: 1.8307802
time: 4.79 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8341735, upper bound: 1.8350936
time: 4.92 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -9.5384808, -5.6820860, -9.5810766, -5.6971631, -3.5291348, 3.7128835
1: -13.3064976, -8.7983084, -13.2450943, -8.7211609, -4.0701504, 4.0765667
2: -8.1014614, -4.3408709, -8.1438093, -4.2204795, -3.8809819, 3.8029385
3: -9.7938185, -5.0867329, -9.8193684, -5.0897179, -4.5563936, 4.5638423
4: -11.1180458, -7.0958443, -11.1140585, -7.0736213, -4.0444245, 4.0182142
5: -0.2681775, 3.2897587, -0.3077166, 3.2029119, -3.4455023, 3.5206959
6: 4.4577813, 7.5338411, 4.4198332, 7.5251832, -3.0674019, 3.1140079
7: -18.0327682, -14.3295116, -18.0536118, -14.2619324, -3.5257134, 3.4943061
8: 0.0280328, 4.0848241, 0.0292202, 4.1181693, -4.0482073, 3.9555745
9: -8.8882856, -5.6778417, -8.9662056, -5.7122622, -3.0384088, 3.0733085

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8384715, upper bound: 1.8278829
time: 5.36 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8384715, upper bound: 1.8350850
time: 5.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5346842, -5.7105408, -9.5741148, -5.6981163, -3.5151443, 3.6830664
1: -13.1846390, -8.7899284, -13.2418213, -8.7624750, -4.0004740, 4.0643716
2: -8.0927334, -4.3504043, -8.1350327, -4.2243090, -3.8684244, 3.7846284
3: -9.7975969, -5.1735311, -9.7900743, -5.0928912, -4.5574265, 4.4613023
4: -11.0601025, -7.0887427, -11.1105328, -7.0972643, -3.9628382, 4.0217900
5: -0.2551088, 3.1854455, -0.2690077, 3.2017012, -3.4284554, 3.4494977
6: 4.4715533, 7.5042758, 4.4295406, 7.5230298, -3.0514765, 3.0747352
7: -18.0248890, -14.3027792, -18.0528088, -14.2622004, -3.5217776, 3.5128274
8: 0.0937289, 4.0807047, 0.0340750, 4.0976624, -3.9784336, 3.9408417
9: -8.8935966, -5.7405100, -8.9420567, -5.7165127, -3.0415754, 3.0255458

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8361720, upper bound: 1.8307803
time: 4.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8361720, upper bound: 1.8307806
time: 4.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.5346775, -5.7105412, -9.6025152, -5.6596270, -3.5571680, 3.7234349
1: -13.1846352, -8.7899475, -13.3932991, -8.7005768, -4.0636530, 4.1403971
2: -8.0927267, -4.3504081, -8.1680317, -4.1959810, -3.8967457, 3.8176236
3: -9.7975845, -5.1735344, -9.8324986, -4.9900002, -4.6332788, 4.5042911
4: -11.0600967, -7.0887675, -11.1979847, -7.0555687, -4.0045280, 4.1092172
5: -0.2550809, 3.1854451, -0.3404851, 3.3167906, -3.4905334, 3.5237641
6: 4.4715681, 7.5042748, 4.3800750, 7.5635338, -3.0919657, 3.1241999
7: -18.0248871, -14.3027897, -18.0867386, -14.2404499, -3.5459995, 3.5471191
8: 0.0937343, 4.0806932, -0.0540061, 4.1352849, -4.0167675, 3.9979854
9: -8.8935823, -5.7405148, -8.9756489, -5.6257110, -3.1088209, 3.0602918

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404698, upper bound: 1.8278827
time: 5.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404699, upper bound: 1.8350848
time: 5.37 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.5810862, -5.6971622, -9.5106325, -5.7205982, -3.6697693, 3.4964061
1: -13.2450981, -8.7211409, -13.1549025, -8.8600349, -4.0143189, 4.0151138
2: -8.1438169, -4.2204733, -8.0685768, -4.3693171, -3.7744999, 3.8481035
3: -9.8193817, -5.0897155, -9.7514420, -5.1897492, -4.4721804, 4.5132842
4: -11.1140604, -7.0735970, -11.0311842, -7.1375422, -3.9765182, 3.9575872
5: -0.3077459, 3.2029142, -0.1968565, 3.1748157, -3.4762268, 3.3710632
6: 4.4198179, 7.5251837, 4.5073214, 7.4933276, -3.0735097, 3.0178623
7: -18.0536118, -14.2619219, -17.9988384, -14.3511667, -3.4698963, 3.5014582
8: 0.0292166, 4.1181798, 0.1162175, 4.0473661, -3.9169569, 3.9692836
9: -8.9662170, -5.7122583, -8.8550072, -5.7684307, -3.0250077, 3.0040097

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307805, upper bound: 1.8341731
time: 4.79 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307805, upper bound: 1.8341737
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5810766, -5.6971631, -9.5384808, -5.6820860, -3.7128835, 3.5291352
1: -13.2450943, -8.7211609, -13.3064976, -8.7983084, -4.0765667, 4.0701504
2: -8.1438093, -4.2204795, -8.1014614, -4.3408709, -3.8029385, 3.8809819
3: -9.8193684, -5.0897179, -9.7938185, -5.0867329, -4.5638423, 4.5563936
4: -11.1140585, -7.0736213, -11.1180458, -7.0958443, -4.0182142, 4.0444245
5: -0.3077166, 3.2029119, -0.2681775, 3.2897587, -3.5206962, 3.4455018
6: 4.4198332, 7.5251832, 4.4577813, 7.5338411, -3.1140079, 3.0674019
7: -18.0536118, -14.2619324, -18.0327682, -14.3295116, -3.4943066, 3.5257139
8: 0.0292202, 4.1181693, 0.0280328, 4.0848241, -3.9555731, 4.0482078
9: -8.9662056, -5.7122622, -8.8882856, -5.6778417, -3.0733085, 3.0384088

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278833, upper bound: 1.8384712
time: 4.76 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350855, upper bound: 1.8384712
time: 5.13 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.5810862, -5.6971622, -9.5577078, -5.7061796, -3.6875296, 3.6761398
1: -13.2450981, -8.7211409, -13.2175655, -8.7850094, -4.0462136, 4.0778546
2: -8.1438169, -4.2204733, -8.1210327, -4.2373271, -3.9064898, 3.9005594
3: -9.8193817, -5.0897155, -9.7743893, -5.1045084, -4.5932837, 4.5636306
4: -11.1140604, -7.0735970, -11.0869722, -7.1183386, -3.9957218, 4.0133753
5: -0.3077459, 3.2029142, -0.2526393, 3.1930523, -3.4825315, 3.4373269
6: 4.4198179, 7.5251837, 4.4506121, 7.5146208, -3.0948029, 3.0745716
7: -18.0536118, -14.2619219, -18.0293694, -14.3005600, -3.5166531, 3.5355635
8: 0.0292166, 4.1181798, 0.0498326, 4.0859141, -4.0054579, 4.0131769
9: -8.9662170, -5.7122583, -8.9291229, -5.7357955, -3.0364046, 3.0228329

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8341738
time: 5.07 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8341742
time: 4.95 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -9.5810766, -5.6971631, -9.5860157, -5.6676865, -3.7306690, 3.7165537
1: -13.2450943, -8.7211609, -13.3690300, -8.7230701, -4.1092048, 4.1361628
2: -8.1438093, -4.2204795, -8.1539984, -4.2089815, -3.9348278, 3.9335189
3: -9.8193684, -5.0897179, -9.8168030, -5.0016189, -4.6563978, 4.6066895
4: -11.1140585, -7.0736213, -11.1744175, -7.0765972, -4.0374613, 4.1007962
5: -0.3077166, 3.2029119, -0.3242118, 3.3080959, -3.5300741, 3.5117517
6: 4.4198332, 7.5251832, 4.4010963, 7.5551248, -3.1352916, 3.1240869
7: -18.0536118, -14.2619324, -18.0633087, -14.2787580, -3.5411377, 3.5578456
8: 0.0292202, 4.1181693, -0.0382140, 4.1234989, -4.0437899, 4.0815330
9: -8.9662056, -5.7122622, -8.9627171, -5.6450510, -3.1002769, 3.0576193

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278804, upper bound: 1.8384713
time: 5.02 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350825, upper bound: 1.8384717
time: 5.38 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5741148, -5.6981163, -9.5346842, -5.7105408, -3.6830664, 3.5151439
1: -13.2418213, -8.7624750, -13.1846390, -8.7899284, -4.0643721, 4.0004740
2: -8.1350327, -4.2243090, -8.0927334, -4.3504043, -3.7846284, 3.8684244
3: -9.7900743, -5.0928912, -9.7975969, -5.1735311, -4.4613018, 4.5574269
4: -11.1105328, -7.0972643, -11.0601025, -7.0887427, -4.0217900, 3.9628382
5: -0.2690077, 3.2017012, -0.2551088, 3.1854455, -3.4494972, 3.4284549
6: 4.4295406, 7.5230298, 4.4715533, 7.5042758, -3.0747352, 3.0514765
7: -18.0528088, -14.2622004, -18.0248890, -14.3027792, -3.5128288, 3.5217776
8: 0.0340750, 4.0976624, 0.0937289, 4.0807047, -3.9408417, 3.9784336
9: -8.9420567, -5.7165127, -8.8935966, -5.7405100, -3.0255461, 3.0415759

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307804, upper bound: 1.8361718
time: 4.71 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307804, upper bound: 1.8404773
time: 4.97 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -9.6025152, -5.6596270, -9.5346775, -5.7105412, -3.7234349, 3.5571685
1: -13.3932991, -8.7005768, -13.1846352, -8.7899475, -4.1403961, 4.0636530
2: -8.1680317, -4.1959810, -8.0927267, -4.3504081, -3.8176236, 3.8967457
3: -9.8324986, -4.9900002, -9.7975845, -5.1735344, -4.5042915, 4.6332788
4: -11.1979847, -7.0555687, -11.0600967, -7.0887675, -4.1092172, 4.0045280
5: -0.3404851, 3.3167906, -0.2550809, 3.1854451, -3.5237646, 3.4905336
6: 4.3800750, 7.5635338, 4.4715681, 7.5042748, -3.1241999, 3.0919657
7: -18.0867386, -14.2404499, -18.0248871, -14.3027897, -3.5471191, 3.5459995
8: -0.0540061, 4.1352849, 0.0937343, 4.0806932, -3.9979854, 4.0167675
9: -8.9756489, -5.6257110, -8.8935823, -5.7405148, -3.0602918, 3.1088204

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278830, upper bound: 1.8404695
time: 4.87 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350851, upper bound: 1.8404696
time: 4.96 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5741148, -5.6981163, -9.5826015, -5.6961126, -3.7007852, 3.6948204
1: -13.2418213, -8.7624750, -13.2471256, -8.7146549, -4.0982771, 4.0664902
2: -8.1350327, -4.2243090, -8.1452131, -4.2182064, -3.9168262, 3.9209042
3: -9.7900743, -5.0928912, -9.8206444, -5.0881262, -4.5824232, 4.6078358
4: -11.1105328, -7.0972643, -11.1171818, -7.0695896, -4.0409431, 4.0199175
5: -0.2690077, 3.2017012, -0.3109784, 3.2038751, -3.4555016, 3.4947281
6: 4.4295406, 7.5230298, 4.4145718, 7.5255613, -3.0960207, 3.1084580
7: -18.0528088, -14.2622004, -18.0553322, -14.2519817, -3.5602245, 3.5538287
8: 0.0340750, 4.0976624, 0.0272675, 4.1195230, -4.0334949, 4.0223904
9: -8.9420567, -5.7165127, -8.9681683, -5.7078476, -3.0372748, 3.0605366

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8361723
time: 5.00 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8404778
time: 5.05 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.6025152, -5.6596270, -9.5825949, -5.6961136, -3.7411523, 3.7379804
1: -13.3932991, -8.7005768, -13.2471199, -8.7146740, -4.1579876, 4.1296673
2: -8.1680317, -4.1959810, -8.1452065, -4.2182117, -3.9498200, 3.9492254
3: -9.8324986, -4.9900002, -9.8206329, -5.0881290, -4.6255054, 4.6661620
4: -11.1979847, -7.0555687, -11.1171780, -7.0696173, -4.1283674, 4.0616093
5: -0.3404851, 3.3167906, -0.3109500, 3.2038746, -3.5298567, 3.5418661
6: 4.3800750, 7.5635338, 4.4145870, 7.5255604, -3.1454854, 3.1489468
7: -18.0867386, -14.2404499, -18.0553284, -14.2519941, -3.5831594, 3.5780511
8: -0.0540061, 4.1352849, 0.0272714, 4.1195116, -4.0986366, 4.0606785
9: -8.9756489, -5.6257110, -8.9681559, -5.7078528, -3.0720630, 3.1210091

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350824, upper bound: 1.8332752
time: 4.80 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350824, upper bound: 1.8404699
time: 4.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.34 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8287834
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8287837
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8278850, upper bound: 1.8330882
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350871, upper bound: 1.8330883
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8307818
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307821, upper bound: 1.8350949
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350871, upper bound: 1.8278848
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350870, upper bound: 1.8350868
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8341735, upper bound: 1.8307802
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8341735, upper bound: 1.8350936
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8384715, upper bound: 1.8278829
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8384715, upper bound: 1.8350850
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8361720, upper bound: 1.8307803
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8361720, upper bound: 1.8307806
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8404698, upper bound: 1.8278827
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8404699, upper bound: 1.8350848
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307805, upper bound: 1.8341731
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307805, upper bound: 1.8341737
IS_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8278833, upper bound: 1.8384712
IS_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350855, upper bound: 1.8384712
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8341738
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8341742
IS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8278804, upper bound: 1.8384713
IS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350825, upper bound: 1.8384717
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307804, upper bound: 1.8361718
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307804, upper bound: 1.8404773
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8278830, upper bound: 1.8404695
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350851, upper bound: 1.8404696
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8361723
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8307775, upper bound: 1.8404778
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350824, upper bound: 1.8332752
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.34
Output dim: 6, lower bound: -1.8350824, upper bound: 1.8404699

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.5247374, -5.7135901, -9.5106325, -5.7205982, -3.4835458, 3.4779124
1: -13.1771851, -8.8441830, -13.1549025, -8.8600349, -3.9439864, 3.9336796
2: -8.0811443, -4.3587360, -8.0685768, -4.3693171, -3.7118273, 3.7098408
3: -9.7657537, -5.1798353, -9.7514420, -5.1897492, -4.4118977, 4.4073696
4: -11.0504217, -7.1204743, -11.0311842, -7.1375422, -3.9128795, 3.9107099
5: -0.2099662, 3.1823218, -0.1968565, 3.1748157, -3.3632469, 3.3578300
6: 4.4915943, 7.5013418, 4.5073214, 7.4933276, -3.0017333, 2.9940205
7: -18.0206184, -14.3229008, -17.9988384, -14.3511667, -3.4380178, 3.4451222
8: 0.1024573, 4.0575089, 0.1162175, 4.0473661, -3.8471909, 3.8385835
9: -8.8656349, -5.7535772, -8.8550072, -5.7684307, -2.9575033, 2.9633443

Time for backsubstitution: 14.55 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.050609588623047
rel_dist={6: [-1.840513682283781, 1.840511023435588]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733650, upper bound: 1.4762265
time: 5.19 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762276, upper bound: 1.4762298
time: 5.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.00 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 11.00
Output dim: 6, lower bound: -1.4733650, upper bound: 1.4762265
IS_B2, status: Status.UNKNOWN, split count: 1, time: 11.00
Output dim: 6, lower bound: -1.4762276, upper bound: 1.4762298

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -9.5391273, -5.7096615, -9.5346851, -5.7105403, -3.3993778, 3.2718325
1: -13.1964693, -8.7866516, -13.1846399, -8.7899218, -3.7311640, 3.7073221
2: -8.1096706, -4.3441396, -8.0927343, -4.3504028, -3.7592678, 3.7485948
3: -9.7992125, -5.1697154, -9.7975979, -5.1735291, -4.1463957, 4.1493454
4: -11.0641098, -7.0842161, -11.0601044, -7.0887394, -3.8447809, 3.7885790
5: -0.2584014, 3.1896486, -0.2551122, 3.1854463, -3.2544775, 3.2482481
6: 4.4682822, 7.5089655, 4.4715471, 7.5042758, -3.0359936, 3.0374184
7: -18.0321293, -14.2988148, -18.0248909, -14.3027716, -3.1878366, 3.1737466
8: 0.0909293, 4.0889697, 0.0937277, 4.0807061, -3.7291603, 3.8030720
9: -8.8969088, -5.7304850, -8.8935957, -5.7405071, -2.7789946, 2.7940686

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4733628
time: 5.15 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4762266
time: 5.24 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -9.5452538, -5.7085781, -9.5826025, -5.6961126, -3.4231191, 3.4507151
1: -13.2110844, -8.7825794, -13.2471294, -8.7146473, -3.7819490, 3.7831392
2: -8.1305313, -4.3364458, -8.1452150, -4.2182059, -3.9123254, 3.8087692
3: -9.8012619, -5.1651192, -9.8206444, -5.0881243, -4.2722149, 4.1857715
4: -11.0695400, -7.0786147, -11.1171827, -7.0695877, -3.8679600, 3.9085512
5: -0.2625172, 3.1949437, -0.3109841, 3.2038767, -3.2690496, 3.3238344
6: 4.4642472, 7.5148230, 4.4145656, 7.5255613, -3.0613141, 3.1002574
7: -18.0411034, -14.2939587, -18.0553322, -14.2519770, -3.2420168, 3.2077932
8: 0.0874968, 4.0993204, 0.0272651, 4.1195259, -3.8355579, 3.8702412
9: -8.9012566, -5.7181067, -8.9681721, -5.7078457, -2.8075495, 2.8593011

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5860

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750562, upper bound: 1.4762262
time: 5.51 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762266, upper bound: 1.4762263
time: 7.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.69 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 27.69
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4733628
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 27.69
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4762266
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 27.69
Output dim: 6, lower bound: -1.4750562, upper bound: 1.4762262
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 27.69
Output dim: 6, lower bound: -1.4762266, upper bound: 1.4762263

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105403, -9.5346851, -5.7105403, -3.2698717, 3.2698722
1: -13.1846399, -8.7899218, -13.1846399, -8.7899218, -3.7172318, 3.7172318
2: -8.0927343, -4.3504028, -8.0927343, -4.3504028, -3.7423315, 3.7423315
3: -9.7975979, -5.1735291, -9.7975979, -5.1735291, -4.1416512, 4.1416512
4: -11.0601044, -7.0887394, -11.0601044, -7.0887394, -3.7828684, 3.7828679
5: -0.2551122, 3.1854463, -0.2551122, 3.1854463, -3.2511034, 3.2511034
6: 4.4715471, 7.5042758, 4.4715471, 7.5042758, -3.0327287, 3.0327287
7: -18.0248909, -14.3027716, -18.0248909, -14.3027716, -3.1803045, 3.1803041
8: 0.0937277, 4.0807061, 0.0937277, 4.0807061, -3.7269478, 3.7269473
9: -8.8935957, -5.7405071, -8.8935957, -5.7405071, -2.7842879, 2.7842882

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4721916
time: 5.08 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733641, upper bound: 1.4733619
time: 5.15 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961126, -9.5346851, -5.7105403, -3.4433126, 3.2852778
1: -13.2471294, -8.7146473, -13.1846399, -8.7899218, -3.7813158, 3.7505178
2: -8.1452150, -4.2182059, -8.0927343, -4.3504028, -3.7948122, 3.8745284
3: -9.8206444, -5.0881243, -9.7975979, -5.1735291, -4.1685448, 4.2400231
4: -11.1171827, -7.0695877, -11.0601044, -7.0887394, -3.8981857, 3.8044596
5: -0.3109841, 3.2038767, -0.2551122, 3.1854463, -3.3217993, 3.2634401
6: 4.4145656, 7.5255613, 4.4715471, 7.5042758, -3.0897102, 3.0540142
7: -18.0553322, -14.2519770, -18.0248909, -14.3027716, -3.2102270, 3.2230077
8: 0.0272651, 4.1195259, 0.0937277, 4.0807061, -3.7762947, 3.8345709
9: -8.9681721, -5.7078457, -8.8935957, -5.7405071, -2.8366013, 2.8151290

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4750553
time: 5.15 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733641, upper bound: 1.4762255
time: 5.42 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -9.5288286, -5.7166457, -9.5800190, -5.6978889, -3.4025550, 3.4362550
1: -13.1868153, -8.8051023, -13.2436733, -8.7256413, -3.7420297, 3.7568097
2: -8.1165867, -4.3494368, -8.1428509, -4.2220526, -3.8945341, 3.7934141
3: -9.7856083, -5.1766095, -9.8184929, -5.0908256, -4.2513819, 4.1694851
4: -11.0459623, -7.0996809, -11.1119232, -7.0763454, -3.8434067, 3.8822298
5: -0.2461469, 3.1862769, -0.3055015, 3.2022395, -3.2506962, 3.3092089
6: 4.4851971, 7.5064096, 4.4234600, 7.5249157, -3.0397186, 3.0829496
7: -18.0175915, -14.3323250, -18.0524330, -14.2687969, -3.1989284, 3.1656556
8: 0.1031973, 4.0875902, 0.0305821, 4.1172462, -3.8112411, 3.8523145
9: -8.8883562, -5.7373796, -8.9648657, -5.7153001, -2.7854424, 2.8335569

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750546, upper bound: 1.4752234
time: 4.84 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750546, upper bound: 1.4762247
time: 5.55 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -9.5452528, -5.7085791, -9.5826025, -5.6961117, -3.4157076, 3.4537859
1: -13.2110853, -8.7825871, -13.2471275, -8.7146511, -3.7726460, 3.7680221
2: -8.1305294, -4.3364482, -8.1452131, -4.2182064, -3.9123230, 3.8087649
3: -9.8012619, -5.1651216, -9.8206444, -5.0881238, -4.2706861, 4.1838703
4: -11.0695391, -7.0786195, -11.1171837, -7.0695877, -3.8679552, 3.9084911
5: -0.2625134, 3.1949439, -0.3109801, 3.2038760, -3.2688909, 3.3233671
6: 4.4642506, 7.5148220, 4.4145699, 7.5255623, -3.0613117, 3.1002522
7: -18.0411015, -14.2939644, -18.0553322, -14.2519798, -3.2292099, 3.1971960
8: 0.0874996, 4.0993214, 0.0272672, 4.1195240, -3.8351345, 3.8607340
9: -8.9012547, -5.7181101, -8.9681692, -5.7078481, -2.8075447, 2.8533144

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762250, upper bound: 1.4752242
time: 6.55 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762250, upper bound: 1.4762256
time: 5.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.41 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4721916
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4733641, upper bound: 1.4733619
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4733640, upper bound: 1.4750553
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4733641, upper bound: 1.4762255
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4750546, upper bound: 1.4752234
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4750546, upper bound: 1.4762247
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4762250, upper bound: 1.4752242
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.41
Output dim: 6, lower bound: -1.4762250, upper bound: 1.4762256

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5321064, -5.7123137, -9.5190468, -5.7185955, -3.2573586, 3.2522788
1: -13.1811895, -8.8009052, -13.1603251, -8.8122578, -3.6919804, 3.6794147
2: -8.0903568, -4.3542576, -8.0787745, -4.3632579, -3.7270989, 3.7245169
3: -9.7954483, -5.1762252, -9.7820282, -5.1850348, -4.1252995, 4.1207905
4: -11.0548038, -7.0954981, -11.0377274, -7.1098089, -3.7577095, 3.7623901
5: -0.2496338, 3.1838012, -0.2387795, 3.1769717, -3.2365465, 3.2329402
6: 4.4803624, 7.5036306, 4.4925060, 7.4958830, -3.0155206, 3.0111246
7: -18.0219593, -14.3196039, -18.0013733, -14.3409834, -3.1381102, 3.1395373
8: 0.0970269, 4.0784407, 0.1094306, 4.0692348, -3.7105751, 3.7024417
9: -8.8903227, -5.7479467, -8.8810453, -5.7597656, -2.7584362, 2.7628460

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733650, upper bound: 1.4711928
time: 6.23 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733650, upper bound: 1.4721937
time: 5.29 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105398, -9.5346842, -5.7105408, -3.2699909, 3.2674003
1: -13.1846390, -8.7899256, -13.1846390, -8.7899284, -3.7009573, 3.7172279
2: -8.0927334, -4.3504033, -8.0927334, -4.3504043, -3.7423291, 3.7423301
3: -9.7975969, -5.1735301, -9.7975969, -5.1735311, -4.1395845, 4.1402864
4: -11.0601044, -7.0887403, -11.0601025, -7.0887427, -3.7807198, 3.7891340
5: -0.2551103, 3.1854451, -0.2551088, 3.1854455, -3.2520590, 3.2507710
6: 4.4715481, 7.5042753, 4.4715533, 7.5042758, -3.0327277, 3.0327220
7: -18.0248928, -14.3027802, -18.0248890, -14.3027792, -3.1688271, 3.1803012
8: 0.0937283, 4.0807052, 0.0937289, 4.0807047, -3.7241678, 3.7255926
9: -8.8935957, -5.7405095, -8.8935966, -5.7405100, -2.7827768, 2.7842844

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723643, upper bound: 1.4733639
time: 5.25 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733649, upper bound: 1.4733639
time: 5.63 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.5800190, -5.6978889, -9.5190468, -5.7185955, -3.4288464, 3.2676940
1: -13.2436733, -8.7256413, -13.1603251, -8.8122578, -3.7559547, 3.7106297
2: -8.1428509, -4.2220526, -8.0787745, -4.3632579, -3.7795930, 3.8567219
3: -9.8184929, -5.0908256, -9.7820282, -5.1850348, -4.1522779, 4.2191534
4: -11.1119232, -7.0763454, -11.0377274, -7.1098089, -3.8718586, 3.7839746
5: -0.3055015, 3.2022395, -0.2387795, 3.1769717, -3.3072729, 3.2451315
6: 4.4234600, 7.5249157, 4.4925060, 7.4958830, -3.0724230, 3.0324097
7: -18.0524330, -14.2687969, -18.0013733, -14.3409834, -3.1680546, 3.1819196
8: 0.0305821, 4.1172462, 0.1094306, 4.0692348, -3.7595096, 3.8102493
9: -8.9648657, -5.7153001, -8.8810453, -5.7597656, -2.8109365, 2.7936692

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723617, upper bound: 1.4750537
time: 5.11 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733624, upper bound: 1.4750533
time: 5.11 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961117, -9.5346842, -5.7105408, -3.4463825, 3.2828040
1: -13.2471275, -8.7146511, -13.1846390, -8.7899284, -3.7650414, 3.7412126
2: -8.1452131, -4.2182064, -8.0927334, -4.3504043, -3.7948089, 3.8745270
3: -9.8206444, -5.0881238, -9.7975969, -5.1735311, -4.1666718, 4.2385817
4: -11.1171837, -7.0695877, -11.0601025, -7.0887427, -3.8981247, 3.8107243
5: -0.3109801, 3.2038760, -0.2551088, 3.1854455, -3.3211508, 3.2632852
6: 4.4145699, 7.5255623, 4.4715533, 7.5042758, -3.0897059, 3.0540090
7: -18.0553322, -14.2519798, -18.0248890, -14.3027792, -3.1987495, 3.2122149
8: 0.0272672, 4.1195240, 0.0937289, 4.0807047, -3.7684937, 3.8341446
9: -8.9681692, -5.7078481, -8.8935966, -5.7405100, -2.8306155, 2.8151257

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723617, upper bound: 1.4762240
time: 5.32 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733624, upper bound: 1.4762235
time: 5.17 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5253363, -5.7174501, -9.5715294, -5.6998954, -3.3961320, 3.4271960
1: -13.1846046, -8.8246927, -13.2383547, -8.7734671, -3.6924458, 3.7312131
2: -8.1123943, -4.3519225, -8.1326418, -4.2281523, -3.8842421, 3.7807193
3: -9.7730637, -5.1785293, -9.7879219, -5.0955877, -4.2309437, 4.1340837
4: -11.0433149, -7.1110396, -11.1052999, -7.1040230, -3.8081818, 3.8586292
5: -0.2289655, 3.1854043, -0.2635262, 3.2000630, -3.2310381, 3.2660785
6: 4.4912872, 7.5053778, 4.4384174, 7.5223832, -3.0310960, 3.0669603
7: -18.0165691, -14.3365192, -18.0499096, -14.2790203, -3.1847067, 3.1581836
8: 0.1059612, 4.0786190, 0.0373878, 4.0953851, -3.7774038, 3.8256545
9: -8.8776693, -5.7407746, -8.9387484, -5.7239656, -2.7647524, 2.8019824

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4740539, upper bound: 1.4752238
time: 5.24 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4740539, upper bound: 1.4752232
time: 5.21 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5288143, -5.7166491, -9.5986414, -5.6616869, -3.4418926, 3.4656334
1: -13.1868076, -8.8051357, -13.3793850, -8.7120419, -3.7532797, 3.8060522
2: -8.1165752, -4.3494430, -8.1642666, -4.2000346, -3.9165406, 3.8148236
3: -9.7855864, -5.1766143, -9.8299913, -5.0012202, -4.2907677, 4.1798210
4: -11.0459566, -7.0997210, -11.1870451, -7.0629101, -3.8578863, 3.9237542
5: -0.2461002, 3.1862764, -0.3334718, 3.3082633, -3.3055744, 3.3374953
6: 4.4852204, 7.5064058, 4.3898916, 7.5623260, -3.0771055, 3.1165142
7: -18.0175858, -14.3323402, -18.0827770, -14.2578392, -3.2086363, 3.1963301
8: 0.1032041, 4.0875692, -0.0444312, 4.1323814, -3.8205214, 3.8865461
9: -8.8883352, -5.7373857, -8.9721127, -5.6394658, -2.8392344, 2.8401866

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750326, upper bound: 1.4728588
time: 6.45 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750508, upper bound: 1.4762205
time: 5.30 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5417519, -5.7093849, -9.5741158, -5.6981201, -3.4092913, 3.4447279
1: -13.2088757, -8.8021755, -13.2418213, -8.7624779, -3.7230611, 3.7424450
2: -8.1263371, -4.3389354, -8.1350307, -4.2243109, -3.9020262, 3.7960954
3: -9.7887192, -5.1670451, -9.7900734, -5.0928917, -4.2492151, 4.1484861
4: -11.0668840, -7.0899744, -11.1105328, -7.0972648, -3.8327236, 3.8848839
5: -0.2453408, 3.1940694, -0.2690063, 3.2017016, -3.2492533, 3.2803848
6: 4.4703522, 7.5137901, 4.4295411, 7.5230293, -3.0526772, 3.0842490
7: -18.0400791, -14.2981586, -18.0528088, -14.2622023, -3.2149854, 3.1897244
8: 0.0902655, 4.0903563, 0.0340769, 4.0976624, -3.8012714, 3.8340826
9: -8.8905716, -5.7215061, -8.9420567, -5.7165146, -2.7868586, 2.8217394

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4752243, upper bound: 1.4752240
time: 5.22 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4752243, upper bound: 1.4752233
time: 5.11 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.5452394, -5.7085810, -9.6012774, -5.6599102, -3.4550519, 3.4830704
1: -13.2110786, -8.7826195, -13.3828487, -8.7010517, -3.7838964, 3.8158488
2: -8.1305199, -4.3364563, -8.1667061, -4.1961937, -3.9343262, 3.8302498
3: -9.8012447, -5.1651263, -9.8321505, -4.9985170, -4.3063750, 4.1942143
4: -11.0695324, -7.0786591, -11.1922598, -7.0561609, -3.8824186, 3.9458809
5: -0.2624662, 3.1949401, -0.3389406, 3.3099055, -3.3214226, 3.3521256
6: 4.4642758, 7.5148177, 4.3809986, 7.5629745, -3.0986986, 3.1338191
7: -18.0410957, -14.2939835, -18.0856705, -14.2410316, -3.2389040, 3.2278748
8: 0.0875050, 4.0993009, -0.0477471, 4.1346836, -3.8443794, 3.8949618
9: -8.9012318, -5.7181168, -8.9754333, -5.6320095, -2.8588996, 2.8599508

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4761972, upper bound: 1.4728588
time: 5.67 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762212, upper bound: 1.4762207
time: 4.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.22 seconds
IS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4733650, upper bound: 1.4711928
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4733650, upper bound: 1.4721937
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4723643, upper bound: 1.4733639
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4733649, upper bound: 1.4733639
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4723617, upper bound: 1.4750537
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4733624, upper bound: 1.4750533
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4723617, upper bound: 1.4762240
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4733624, upper bound: 1.4762235
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4740539, upper bound: 1.4752238
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4740539, upper bound: 1.4752232
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4750326, upper bound: 1.4728588
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4750508, upper bound: 1.4762205
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4752243, upper bound: 1.4752240
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4752243, upper bound: 1.4752233
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4761972, upper bound: 1.4728588
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.22
Output dim: 6, lower bound: -1.4762212, upper bound: 1.4762207

## BFS IS instance: IS_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.5286379, -5.7131157, -9.5106325, -5.7205982, -3.2514157, 3.2442269
1: -13.1789761, -8.8204842, -13.1549025, -8.8600349, -3.6422896, 3.6536937
2: -8.0861626, -4.3567286, -8.0685768, -4.3693171, -3.7168455, 3.7118483
3: -9.7829075, -5.1781363, -9.7514420, -5.1897492, -4.1045017, 4.0852623
4: -11.0522003, -7.1068602, -11.0311842, -7.1375422, -3.7237015, 3.7398758
5: -0.2324502, 3.1829348, -0.1968565, 3.1748157, -3.2168655, 3.1896873
6: 4.4864616, 7.5025973, 4.5073214, 7.4933276, -3.0068660, 2.9952760
7: -18.0209370, -14.3237896, -17.9988384, -14.3511667, -3.1237602, 3.1320405
8: 0.0997825, 4.0694804, 0.1162175, 4.0473661, -3.6760921, 3.6766219
9: -8.8796616, -5.7513428, -8.8550072, -5.7684307, -2.7367392, 2.7300000

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723642, upper bound: 1.4711928
time: 7.74 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723642, upper bound: 1.4711942
time: 5.35 seconds

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5320969, -5.7123156, -9.5372887, -5.6823711, -3.2959495, 3.2756805
1: -13.1811848, -8.8009396, -13.2960377, -8.7987804, -3.7024078, 3.7396963
2: -8.0903463, -4.3542662, -8.1001463, -4.3410807, -3.7492657, 3.7458801
3: -9.7954311, -5.1762323, -9.7934704, -5.0952606, -4.2017889, 4.1311908
4: -11.0547962, -7.0955396, -11.1122942, -7.0964403, -3.7701378, 3.8239641
5: -0.2495866, 3.1837983, -0.2666390, 3.2828755, -3.2909899, 3.2610488
6: 4.4803848, 7.5036287, 4.4587412, 7.5332818, -3.0528970, 3.0448875
7: -18.0219574, -14.3196182, -18.0317020, -14.3300877, -3.1479588, 3.1701589
8: 0.0970337, 4.0784187, 0.0342820, 4.0842323, -3.7197676, 3.7551389
9: -8.8903008, -5.7479529, -8.8880711, -5.6839786, -2.8160210, 2.7680354

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 877

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699780, upper bound: 1.4721548
time: 5.36 seconds

## Relational analysis of IS_B1_A1_B1_B2_B2

### Relational analysis result of IS_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733608, upper bound: 1.4721898
time: 5.56 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5262566, -5.7125421, -9.5312157, -5.7113428, -3.2619276, 3.2614665
1: -13.1792183, -8.8376989, -13.1824284, -8.8095074, -3.6752453, 3.6674757
2: -8.0825386, -4.3564653, -8.0885410, -4.3528757, -3.7296629, 3.7320757
3: -9.7670164, -5.1782541, -9.7850571, -5.1754427, -4.1040535, 4.1194816
4: -11.0535564, -7.1164665, -11.0574999, -7.1001062, -3.7581997, 3.7551298
5: -0.2131999, 3.1832900, -0.2379260, 3.1845813, -3.2088256, 3.2310858
6: 4.4864030, 7.5017223, 4.4776597, 7.5032434, -3.0168405, 3.0240626
7: -18.0223541, -14.3129578, -18.0238667, -14.3069668, -3.1613264, 3.1659703
8: 0.1005218, 4.0588446, 0.0964864, 4.0717459, -3.6983309, 3.6911063
9: -8.8675709, -5.7491779, -8.8829365, -5.7439065, -2.7499409, 2.7625883

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723643, upper bound: 1.4723631
time: 5.25 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723643, upper bound: 1.4733639
time: 5.63 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5528889, -5.6743116, -9.5346727, -5.7105422, -3.2931662, 3.3060131
1: -13.3203745, -8.7764759, -13.1846323, -8.7899599, -3.7622013, 3.7275777
2: -8.1141396, -4.3282456, -8.0927210, -4.3504114, -3.7637281, 3.7644753
3: -9.8090534, -5.0837617, -9.7975788, -5.1735363, -4.1499023, 4.2118359
4: -11.1346645, -7.0754042, -11.0600948, -7.0887871, -3.8368711, 3.8015494
5: -0.2828977, 3.2913928, -0.2550626, 3.1854439, -3.2800980, 3.3026724
6: 4.4378328, 7.5416837, 4.4715748, 7.5042729, -3.0664401, 3.0701089
7: -18.0552101, -14.2919292, -18.0248871, -14.3027964, -3.1994562, 3.1902351
8: 0.0185564, 4.0957460, 0.0937365, 4.0806842, -3.7727022, 3.7347422
9: -8.9006233, -5.6647182, -8.8935738, -5.7405167, -2.7879744, 2.8383834

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733156, upper bound: 1.4699777
time: 4.59 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733611, upper bound: 1.4733598
time: 5.23 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5715294, -5.6998954, -9.5155811, -5.7194014, -3.4197898, 3.2616363
1: -13.2383547, -8.7734671, -13.1581154, -8.8318396, -3.7303500, 3.6610587
2: -8.1326418, -4.2281523, -8.0745802, -4.3657279, -3.7669139, 3.8464279
3: -9.7879219, -5.0955877, -9.7694874, -5.1869440, -4.1169062, 4.1984143
4: -11.1052999, -7.1040230, -11.0351248, -7.1211758, -3.8482542, 3.7500286
5: -0.2635262, 3.2000630, -0.2215912, 3.1761055, -3.2640057, 3.2254796
6: 4.4384174, 7.5223832, 4.4985948, 7.4948506, -3.0564332, 3.0237885
7: -18.0499096, -14.2790203, -18.0003510, -14.3451710, -3.1605597, 3.1676965
8: 0.0373878, 4.0953851, 0.1121842, 4.0602717, -3.7320609, 3.7764411
9: -8.9387484, -5.7239656, -8.8703814, -5.7631588, -2.7793612, 2.7719803

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723617, upper bound: 1.4740530
time: 5.14 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723617, upper bound: 1.4750535
time: 4.98 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5986414, -5.6616869, -9.5190344, -5.7185993, -3.4582238, 3.3059688
1: -13.3793850, -8.7120419, -13.1603203, -8.8122892, -3.8084717, 3.7218809
2: -8.1642666, -4.2000346, -8.0787640, -4.3632646, -3.8010020, 3.8787293
3: -9.8299913, -5.0012202, -9.7820101, -5.1850395, -4.1626129, 4.2757277
4: -11.1870451, -7.0629101, -11.0377207, -7.1098518, -3.9133396, 3.7965136
5: -0.3334718, 3.3082633, -0.2387331, 3.1769691, -3.3353472, 3.2968392
6: 4.3898916, 7.5623260, 4.4925308, 7.4958792, -3.1059875, 3.0697951
7: -18.0827770, -14.2578392, -18.0013714, -14.3409996, -3.1987162, 3.1916275
8: -0.0444312, 4.1323814, 0.1094371, 4.0692148, -3.7942109, 3.8195939
9: -8.9721127, -5.6394658, -8.8810234, -5.7597713, -2.8175650, 2.8388529

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699756, upper bound: 1.4750319
time: 5.61 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733582, upper bound: 1.4750499
time: 5.73 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5741158, -5.6981201, -9.5312157, -5.7113428, -3.4373240, 3.2767520
1: -13.2418213, -8.7624779, -13.1824284, -8.8095074, -3.7394085, 3.6916373
2: -8.1350307, -4.2243109, -8.0885410, -4.3528757, -3.7821550, 3.8642302
3: -9.7900734, -5.0928917, -9.7850571, -5.1754427, -4.1313162, 4.2178597
4: -11.1105328, -7.0972648, -11.0574999, -7.1001062, -3.8745146, 3.7767725
5: -0.2690063, 3.2017016, -0.2379260, 3.1845813, -3.2780571, 3.2436523
6: 4.4295411, 7.5230293, 4.4776597, 7.5032434, -3.0737023, 3.0453696
7: -18.0528088, -14.2622023, -18.0238667, -14.3069668, -3.1912632, 3.1979890
8: 0.0340769, 4.0976624, 0.0964864, 4.0717459, -3.7410312, 3.8003221
9: -8.9420567, -5.7165146, -8.8829365, -5.7439065, -2.7990394, 2.7934411

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723618, upper bound: 1.4752229
time: 4.87 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723618, upper bound: 1.4762240
time: 4.90 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.6012774, -5.6599102, -9.5346727, -5.7105422, -3.4756665, 3.3210969
1: -13.3828487, -8.7010517, -13.1846323, -8.7899599, -3.8166370, 3.7524631
2: -8.1667061, -4.1961937, -8.0927210, -4.3504114, -3.8162947, 3.8965273
3: -9.8321505, -4.9985170, -9.7975788, -5.1735363, -4.1770172, 4.2913480
4: -11.1922598, -7.0561609, -11.0600948, -7.0887871, -3.9354668, 3.8232460
5: -0.3389406, 3.3099055, -0.2550626, 3.1854439, -3.3497438, 3.3126287
6: 4.3809986, 7.5629745, 4.4715748, 7.5042729, -3.1232743, 3.0913997
7: -18.0856705, -14.2410316, -18.0248871, -14.3027964, -3.2286458, 3.2219081
8: -0.0477471, 4.1346836, 0.0937365, 4.0806842, -3.8031816, 3.8434572
9: -8.9754333, -5.6320095, -8.8935738, -5.7405167, -2.8372512, 2.8577483

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699756, upper bound: 1.4761966
time: 4.86 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733583, upper bound: 1.4762197
time: 5.26 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.5203400, -5.7186508, -9.5715294, -5.6998954, -3.3915544, 3.4252276
1: -13.1813707, -8.8529053, -13.2383547, -8.7734671, -3.6889372, 3.7034287
2: -8.1063910, -4.3555365, -8.1326418, -4.2281523, -3.8782387, 3.7771053
3: -9.7550163, -5.1813536, -9.7879219, -5.0955877, -4.2118816, 4.1298413
4: -11.0393114, -7.1274028, -11.1052999, -7.1040230, -3.8015556, 3.8404045
5: -0.2042427, 3.1841023, -0.2635262, 3.2000630, -3.2061558, 3.2646747
6: 4.5000162, 7.5038562, 4.4384174, 7.5223832, -3.0223670, 3.0654387
7: -18.0150547, -14.3425236, -18.0499096, -14.2790203, -3.1835632, 3.1502256
8: 0.1100093, 4.0657058, 0.0373878, 4.0953851, -3.7695045, 3.8104739
9: -8.8622627, -5.7460480, -8.9387484, -5.7239656, -2.7482872, 2.7964578

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 4597

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4740256, upper bound: 1.4718326
time: 5.13 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4740567, upper bound: 1.4752193
time: 5.29 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.5458698, -5.6811800, -9.5715294, -5.6998954, -3.4200339, 3.4666476
1: -13.3082209, -8.7923527, -13.2383547, -8.7734671, -3.7249937, 3.7531264
2: -8.1361418, -4.3278193, -8.1326418, -4.2281523, -3.9079895, 3.8048224
3: -9.7965641, -5.0985260, -9.7879219, -5.0955877, -4.2460117, 4.1963568
4: -11.1131182, -7.0871620, -11.1052999, -7.1040230, -3.8691454, 3.8859429
5: -0.2716241, 3.2828560, -0.2635262, 3.2000630, -3.2763033, 3.2943563
6: 4.4531198, 7.5430508, 4.4384174, 7.5223832, -3.0692635, 3.1046333
7: -18.0464706, -14.3222790, -18.0499096, -14.2790203, -3.1967716, 3.1734729
8: 0.0366485, 4.1018443, 0.0373878, 4.0953851, -3.8243055, 3.8392324
9: -8.8951645, -5.6695013, -8.9387484, -5.7239656, -2.7823248, 2.8340106

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4740256, upper bound: 1.4718329
time: 4.82 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4740547, upper bound: 1.4752211
time: 4.89 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5056553, -5.7336292, -9.5898685, -5.6661911, -3.2912164, 3.4364033
1: -13.1274481, -8.8534431, -13.3535995, -8.7164879, -3.6578684, 3.7425592
2: -8.0882769, -4.3893213, -8.1536093, -4.2103291, -3.8779478, 3.7642879
3: -9.7337389, -5.2391014, -9.8237934, -5.0319190, -4.1992307, 4.1063838
4: -10.9605465, -7.1725621, -11.1738739, -7.1028538, -3.5504971, 3.8389006
5: -0.2040052, 3.1664014, -0.3267374, 3.2985427, -3.2483273, 3.3351524
6: 4.5316205, 7.4803381, 4.4005165, 7.5495682, -2.7194724, 3.0798216
7: -18.0009747, -14.3576574, -18.0761261, -14.2637539, -3.1838140, 3.1768670
8: 0.1516894, 4.0400524, -0.0236657, 4.1259308, -3.7639017, 3.7416377
9: -8.8064413, -5.8043094, -8.9660540, -5.6787405, -2.7067394, 2.7084012

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750317, upper bound: 1.4726291
time: 5.03 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750317, upper bound: 1.4728578
time: 5.17 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5288105, -5.7166519, -9.5986376, -5.6616917, -3.4396453, 3.4541001
1: -13.1867914, -8.8051386, -13.3793745, -8.7120447, -3.7078772, 3.7707920
2: -8.1165667, -4.3494530, -8.1642590, -4.2000399, -3.9165268, 3.8148060
3: -9.7855816, -5.1766438, -9.8299866, -5.0012383, -4.2580404, 4.1388021
4: -11.0459480, -7.0997534, -11.1870384, -7.0629306, -3.8381548, 3.8490672
5: -0.2460938, 3.1862662, -0.3334699, 3.3082576, -3.2954621, 3.3230767
6: 4.4852295, 7.5063944, 4.3898973, 7.5623193, -3.0770898, 3.1164970
7: -18.0175800, -14.3323421, -18.0827732, -14.2578421, -3.2018423, 3.1963210
8: 0.1032209, 4.0875640, -0.0444200, 4.1323786, -3.7948895, 3.8614459
9: -8.8883305, -5.7374191, -8.9721098, -5.6394882, -2.7828846, 2.7455189

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 508

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 877

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4724888, upper bound: 1.4755411
time: 5.74 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727486, upper bound: 1.4739204
time: 5.71 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5367470, -5.7105846, -9.5741158, -5.6981201, -3.4047222, 3.4427595
1: -13.2056465, -8.8303823, -13.2418213, -8.7624779, -3.7194524, 3.7146645
2: -8.1203346, -4.3425531, -8.1350307, -4.2243109, -3.8960238, 3.7924776
3: -9.7706738, -5.1698742, -9.7900734, -5.0928917, -4.2312040, 4.1442404
4: -11.0628815, -7.1063337, -11.1105328, -7.0972648, -3.8260813, 3.8666639
5: -0.2206242, 3.1927705, -0.2690063, 3.2017016, -3.2243919, 3.2789836
6: 4.4790969, 7.5122690, 4.4295411, 7.5230293, -3.0439324, 3.0827279
7: -18.0385628, -14.3041668, -18.0528088, -14.2622023, -3.2138391, 3.1817646
8: 0.0943179, 4.0774460, 0.0340769, 4.0976624, -3.7933445, 3.8189039
9: -8.8751717, -5.7267818, -8.9420567, -5.7165146, -2.7703943, 2.8162134

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 4597
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4751872, upper bound: 1.4718326
time: 5.87 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4752250, upper bound: 1.4752199
time: 5.97 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5623093, -5.6731110, -9.5741158, -5.6981201, -3.4332314, 3.4841976
1: -13.3325205, -8.7698631, -13.2418213, -8.7624779, -3.7555137, 3.7628686
2: -8.1501207, -4.3148522, -8.1350307, -4.2243109, -3.9258099, 3.8201785
3: -9.8122330, -5.0870447, -9.7900734, -5.0928917, -4.2616262, 4.2066755
4: -11.1367245, -7.0661268, -11.1105328, -7.0972648, -3.8875360, 3.9121857
5: -0.2879329, 3.2915640, -0.2690063, 3.2017016, -3.2944889, 3.3064547
6: 4.4322200, 7.5514727, 4.4295411, 7.5230293, -3.0908093, 3.1219316
7: -18.0699692, -14.2839718, -18.0528088, -14.2622023, -3.2270608, 3.2049718
8: 0.0209298, 4.1136246, 0.0340769, 4.0976624, -3.8421202, 3.8476872
9: -8.9080696, -5.6502280, -8.9420567, -5.7165146, -2.8044367, 2.8537598

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 877

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4751872, upper bound: 1.4718328
time: 5.02 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4752250, upper bound: 1.4752192
time: 5.19 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5213041, -5.7260046, -9.5924816, -5.6644087, -3.3063726, 3.4529772
1: -13.1510124, -8.8311691, -13.3570671, -8.7054987, -3.6872993, 3.7507222
2: -8.1022215, -4.3769560, -8.1560516, -4.2064910, -3.8957305, 3.7790956
3: -9.7492447, -5.2276316, -9.8259516, -5.0292163, -4.2147717, 4.1207891
4: -10.9811125, -7.1514931, -11.1791286, -7.0961018, -3.5756226, 3.8610220
5: -0.2203693, 3.1744344, -0.3322031, 3.3001835, -3.2642074, 3.3461633
6: 4.5116510, 7.4887323, 4.3916368, 7.5502129, -2.7384748, 3.0970955
7: -18.0245056, -14.3197117, -18.0790195, -14.2469463, -3.2141013, 3.2077341
8: 0.1359984, 4.0514402, -0.0269773, 4.1282234, -3.7877636, 3.7505774
9: -8.8189859, -5.7852859, -8.9693213, -5.6712918, -2.7255912, 2.7283092

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 4597

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4761963, upper bound: 1.4726290
time: 4.89 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4761962, upper bound: 1.4728578
time: 4.85 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.5452356, -5.7085857, -9.6012745, -5.6599131, -3.4528170, 3.4715366
1: -13.2110662, -8.7826214, -13.3828402, -8.7010546, -3.7385602, 3.7805896
2: -8.1305113, -4.3364635, -8.1667013, -4.1961989, -3.9343123, 3.8302379
3: -9.8012381, -5.1651549, -9.8321457, -4.9985375, -4.2736316, 4.1531963
4: -11.0695229, -7.0786901, -11.1922531, -7.0561786, -3.8565226, 3.8712192
5: -0.2624626, 3.1949310, -0.3389378, 3.3098991, -3.3113122, 3.3360684
6: 4.4642839, 7.5148077, 4.3810043, 7.5629673, -3.0986834, 3.1338034
7: -18.0410919, -14.2939873, -18.0856686, -14.2410316, -3.2321291, 3.2278638
8: 0.0875210, 4.0992956, -0.0477371, 4.1346807, -3.8187494, 3.8698354
9: -8.9012280, -5.7181492, -8.9754286, -5.6320314, -2.8025031, 2.7652502

Time for backsubstitution: 14.52 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.050609588623047
rel_dist={6: [-1.4762376626074394, 1.476239686090869]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4597
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4597

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041176, upper bound: 1.2032546
time: 7.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041155, upper bound: 1.2041149
time: 5.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.03
Output dim: 6, lower bound: -1.2041176, upper bound: 1.2032546
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.03
Output dim: 6, lower bound: -1.2041155, upper bound: 1.2041149

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105403, -9.5361977, -5.7102098, -3.1140614, 3.1152110
1: -13.1846399, -8.7899218, -13.1890688, -8.7887259, -3.5161982, 3.5202742
2: -8.0927343, -4.3504028, -8.0990772, -4.3480754, -3.6900744, 3.6942391
3: -9.7975979, -5.1735291, -9.7981834, -5.1720939, -3.9275618, 3.9264874
4: -11.0601044, -7.0887394, -11.0614004, -7.0870552, -3.5690384, 3.5681129
5: -0.2551122, 3.1854463, -0.2563105, 3.1869838, -3.1401167, 3.1397562
6: 4.4715471, 7.5042758, 4.4703245, 7.5060129, -2.9415522, 2.9412909
7: -18.0248909, -14.3027716, -18.0275917, -14.3013096, -2.9719286, 2.9735384
8: 0.0937277, 4.0807061, 0.0926830, 4.0837522, -3.6119623, 3.6091952
9: -8.8935957, -5.7405071, -8.8947783, -5.7367563, -2.6304331, 2.6277406

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041151, upper bound: 1.2029162
time: 12.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2032565
time: 5.91 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.5826025, -5.6961126, -9.5452433, -5.7085810, -3.2886066, 3.2612481
1: -13.2471294, -8.7146473, -13.2110586, -8.7825851, -3.5762410, 3.5777061
2: -8.1452150, -4.2182059, -8.1304922, -4.3364573, -3.7348223, 3.8109894
3: -9.8206444, -5.0881243, -9.8012581, -5.1651235, -3.9683275, 4.0524335
4: -11.1171827, -7.0695877, -11.0695314, -7.0786247, -3.7217960, 3.6805573
5: -0.3109841, 3.2038767, -0.2625122, 3.1949365, -3.2121916, 3.1569672
6: 4.4145656, 7.5255613, 4.4642525, 7.5148144, -3.0042310, 2.9677944
7: -18.0553322, -14.2519770, -18.0410862, -14.2939644, -2.9983263, 3.0277662
8: 0.0272651, 4.1195259, 0.0875002, 4.0993156, -3.7518177, 3.7178383
9: -8.9681721, -5.7078457, -8.9012508, -5.7181263, -2.7114658, 2.6579378

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5860

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2037909
time: 5.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2041144
time: 5.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.19
Output dim: 6, lower bound: -1.2041151, upper bound: 1.2029162
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.19
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2032565
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.19
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2037909
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.19
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2041144

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.5310669, -5.7130156, -9.5205631, -5.7182693, -3.1002831, 3.0969024
1: -13.1797953, -8.8052530, -13.1647596, -8.8110638, -3.4889450, 3.4780788
2: -8.0894432, -4.3557930, -8.0851192, -4.3609324, -3.6737165, 3.6737018
3: -9.7945747, -5.1773071, -9.7826118, -5.1835942, -3.9101973, 3.9046130
4: -11.0527210, -7.0981345, -11.0390253, -7.1081262, -3.5422716, 3.5450249
5: -0.2474594, 3.1831417, -0.2399716, 3.1785095, -3.1238999, 3.1209726
6: 4.4838572, 7.5033665, 4.4912777, 7.4976187, -2.9206891, 2.9180884
7: -18.0208130, -14.3262472, -18.0040741, -14.3395224, -2.9287300, 2.9264178
8: 0.0983520, 4.0775385, 0.1083853, 4.0722847, -3.5940590, 3.5836606
9: -8.8890276, -5.7508726, -8.8822193, -5.7560153, -2.6030450, 2.6033957

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041169, upper bound: 1.2025660
time: 10.27 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041147, upper bound: 1.2029133
time: 10.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.5346851, -5.7105398, -9.5361996, -5.7102113, -3.1136961, 3.1120949
1: -13.1846380, -8.7899265, -13.1890669, -8.7887344, -3.4983778, 3.5202684
2: -8.0927334, -4.3504038, -8.0990753, -4.3480759, -3.6888266, 3.6936073
3: -9.7975950, -5.1735306, -9.7981815, -5.1720948, -3.9247456, 3.9245629
4: -11.0601006, -7.0887403, -11.0613966, -7.0870576, -3.5659122, 3.5736380
5: -0.2551103, 3.1854458, -0.2563076, 3.1869831, -3.1409597, 3.1392779
6: 4.4715514, 7.5042768, 4.4703283, 7.5060134, -2.9415479, 2.9363136
7: -18.0248909, -14.3027802, -18.0275917, -14.3013163, -2.9593601, 2.9735332
8: 0.0937284, 4.0807056, 0.0926836, 4.0837498, -3.6082029, 3.6070995
9: -8.8935947, -5.7405100, -8.8947773, -5.7367573, -2.6287808, 2.6277373

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2032531
time: 7.17 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041148, upper bound: 1.2032550
time: 5.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.5789766, -5.6985903, -9.5288172, -5.7166500, -3.2728014, 3.2400517
1: -13.2422752, -8.7299891, -13.1867886, -8.8051071, -3.5479531, 3.5345137
2: -8.1419239, -4.2235813, -8.1165485, -4.3494468, -3.7179551, 3.7894509
3: -9.8176155, -5.0919051, -9.7856045, -5.1766138, -3.9510288, 4.0301037
4: -11.1098661, -7.0789809, -11.0459585, -7.0996885, -3.6936855, 3.6536942
5: -0.3033276, 3.2015858, -0.2461414, 3.1862717, -3.1954842, 3.1379828
6: 4.4269905, 7.5246525, 4.4852028, 7.5064020, -2.9833083, 2.9445949
7: -18.0513000, -14.2754364, -18.0175743, -14.3323269, -2.9551897, 2.9783673
8: 0.0319136, 4.1163416, 0.1032016, 4.0875831, -3.7324314, 3.6924376
9: -8.9635582, -5.7182312, -8.8883505, -5.7373986, -2.6842127, 2.6329443

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2037904
time: 12.70 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041169, upper bound: 1.2037879
time: 7.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.5826006, -5.6961117, -9.5452433, -5.7085810, -3.2905664, 3.2523732
1: -13.2471256, -8.7146530, -13.2110567, -8.7825928, -3.5596886, 3.5684035
2: -8.1452141, -4.2182055, -8.1304903, -4.3364582, -3.7330332, 3.8082094
3: -9.8206444, -5.0881248, -9.8012581, -5.1651249, -3.9657154, 4.0462065
4: -11.1171818, -7.0695920, -11.0695305, -7.0786285, -3.7217312, 3.6805525
5: -0.3109794, 3.2038753, -0.2625082, 3.1949356, -3.2100396, 3.1567407
6: 4.4145699, 7.5255623, 4.4642591, 7.5148129, -3.0042272, 2.9627161
7: -18.0553341, -14.2519827, -18.0410843, -14.2939701, -2.9867239, 3.0135889
8: 0.0272672, 4.1195235, 0.0875025, 4.0993128, -3.7409468, 3.7164712
9: -8.9681702, -5.7078485, -8.9012489, -5.7181273, -2.7047939, 2.6579325

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2041142
time: 6.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041147, upper bound: 1.2041140
time: 5.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.26 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2041169, upper bound: 1.2025660
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2041147, upper bound: 1.2029133
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2032531
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2041148, upper bound: 1.2032550
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2037904
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2041169, upper bound: 1.2037879
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2041142
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.26
Output dim: 6, lower bound: -1.2041147, upper bound: 1.2041140

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.5244331, -5.7145700, -9.5121460, -5.7202740, -3.0917931, 3.0876884
1: -13.1755238, -8.8426580, -13.1593304, -8.8588457, -3.4368725, 3.4347806
2: -8.0814247, -4.3605280, -8.0749216, -4.3669958, -3.6359506, 3.6366282
3: -9.7706242, -5.1809869, -9.7520266, -5.1883140, -3.8773165, 3.8662586
4: -11.0476542, -7.1198483, -11.0324783, -7.1358566, -3.5044079, 3.5113254
5: -0.2146306, 3.1814635, -0.1980662, 3.1763518, -3.0884352, 3.0768352
6: 4.4954848, 7.5013800, 4.5060940, 7.4950647, -2.9024897, 2.8961062
7: -18.0188389, -14.3342295, -18.0015373, -14.3497047, -2.9136496, 2.9138689
8: 0.1036474, 4.0604205, 0.1151752, 4.0504160, -3.5543289, 3.5471053
9: -8.8686514, -5.7575440, -8.8561745, -5.7646809, -2.5705404, 2.5668371

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5791

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037732, upper bound: 1.2025671
time: 6.61 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037732, upper bound: 1.2025663
time: 5.44 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5310516, -5.7130184, -9.5368671, -5.6849928, -3.1350946, 3.1168895
1: -13.1797867, -8.8053017, -13.2877779, -8.7991142, -3.4947987, 3.5254467
2: -8.0894279, -4.3558035, -8.1047630, -4.3416805, -3.6851192, 3.6896539
3: -9.7945480, -5.1773143, -9.7932091, -5.1048884, -3.9715152, 3.9141698
4: -11.0527124, -7.0981975, -11.1065025, -7.0958724, -3.5518551, 3.5984621
5: -0.2473929, 3.1831372, -0.2639389, 3.2754185, -3.1690412, 3.1435323
6: 4.4838901, 7.5033636, 4.4616156, 7.5340638, -2.9625235, 2.9496627
7: -18.0208111, -14.3262711, -18.0327644, -14.3299837, -2.9371071, 2.9544401
8: 0.0983622, 4.0775099, 0.0420012, 4.0863256, -3.6017962, 3.6220956
9: -8.8889961, -5.7508807, -8.8887463, -5.6890936, -2.6487513, 2.6079462

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5791

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 877

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2022516, upper bound: 1.2017550
time: 5.25 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2025110, upper bound: 1.2013198
time: 7.12 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.5262566, -5.7125430, -9.5295668, -5.7117682, -3.1044674, 3.1036186
1: -13.1792183, -8.8376999, -13.1847992, -8.8261433, -3.4550495, 3.4681306
2: -8.0825396, -4.3564649, -8.0910797, -4.3528175, -3.6517620, 3.6558623
3: -9.7670155, -5.1782532, -9.7742300, -5.1757827, -3.8863935, 3.8916798
4: -11.0535526, -7.1164680, -11.0563259, -7.1087704, -3.5322189, 3.5357738
5: -0.2132001, 3.1832902, -0.2234876, 3.1853075, -3.0968442, 3.1038022
6: 4.4864039, 7.5017223, 4.4819722, 7.5040269, -2.9195504, 2.9181066
7: -18.0223541, -14.3129578, -18.0256176, -14.3093004, -2.9468060, 2.9584732
8: 0.1005213, 4.0588446, 0.0979875, 4.0666342, -3.5716362, 3.5673556
9: -8.8675709, -5.7491775, -8.8743992, -5.7434311, -2.5922365, 2.5952325

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2029087
time: 6.15 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2032531
time: 8.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.5509491, -5.6772647, -9.5361824, -5.7102160, -3.1334729, 3.1469231
1: -13.3076811, -8.7779980, -13.1890583, -8.7887831, -3.5466628, 3.5260510
2: -8.1124115, -4.3311729, -8.0990620, -4.3480864, -3.7048430, 3.7050705
3: -9.8082066, -5.0948315, -9.7981539, -5.1721039, -3.9342289, 3.9811277
4: -11.1275759, -7.0765190, -11.0613871, -7.0871201, -3.6125555, 3.5832138
5: -0.2790132, 3.2823939, -0.2562408, 3.1869795, -3.1634588, 3.1816144
6: 4.4419599, 7.5407295, 4.4703627, 7.5060101, -2.9730744, 2.9755335
7: -18.0535641, -14.2932825, -18.0275860, -14.3013382, -2.9855208, 2.9819965
8: 0.0273256, 4.0947905, 0.0926931, 4.0837221, -3.6413622, 3.6147923
9: -8.9001274, -5.6735902, -8.8947449, -5.7367673, -2.6333404, 2.6709650

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5791

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 877

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029446, upper bound: 1.2013949
time: 7.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2025114, upper bound: 1.2016547
time: 4.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.5704851, -5.7005997, -9.5221415, -5.7182059, -3.2625103, 3.2307267
1: -13.2369518, -8.7778149, -13.1825218, -8.8425388, -3.5047770, 3.4826717
2: -8.1317167, -4.2296829, -8.1085539, -4.3542142, -3.6808643, 3.7516809
3: -9.7870464, -5.0966678, -9.7616453, -5.1803179, -3.9129906, 3.9960194
4: -11.1032467, -7.1066570, -11.0408039, -7.1213984, -3.6585588, 3.6143141
5: -0.2613506, 3.1994061, -0.2133183, 3.1845815, -3.1514721, 3.1025772
6: 4.4419384, 7.5221190, 4.4968214, 7.5044146, -2.9612551, 2.9263506
7: -18.0487766, -14.2856579, -18.0156021, -14.3403244, -2.9426813, 2.9634228
8: 0.0387177, 4.0944805, 0.1085176, 4.0704470, -3.6957388, 3.6536160
9: -8.9374409, -5.7268963, -8.8679247, -5.7440739, -2.6491737, 2.6018443

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2034491
time: 5.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037754, upper bound: 1.2037879
time: 6.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.5955982, -5.6653404, -9.5288019, -5.7166519, -3.2975345, 3.2755470
1: -13.3652906, -8.7179241, -13.1867800, -8.8051548, -3.5843697, 3.5410872
2: -8.1615705, -4.2044911, -8.1165333, -4.3494577, -3.7344956, 3.8018184
3: -9.8282633, -5.0133519, -9.7855749, -5.1766224, -3.9605207, 4.0553846
4: -11.1778679, -7.0666680, -11.0459461, -7.0997515, -3.7251358, 3.6656628
5: -0.3273983, 3.2985961, -0.2460723, 3.1862674, -3.2182908, 3.1839943
6: 4.3975081, 7.5611043, 4.4852366, 7.5063977, -3.0148096, 2.9871931
7: -18.0800018, -14.2658396, -18.0175705, -14.3323526, -2.9832745, 2.9865394
8: -0.0343320, 4.1304779, 0.1032096, 4.0875554, -3.7570176, 3.7002859
9: -8.9702835, -5.6512709, -8.8883200, -5.7374077, -2.6901517, 2.6771157

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5791

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 877

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029446, upper bound: 1.2019359
time: 5.54 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2025114, upper bound: 1.2021923
time: 10.66 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.5741138, -5.6981201, -9.5385523, -5.7101388, -3.2802801, 3.2430611
1: -13.2418213, -8.7624788, -13.2067957, -8.8200169, -3.5165424, 3.5164917
2: -8.1350317, -4.2243099, -8.1224985, -4.3412299, -3.6959348, 3.7704380
3: -9.7900734, -5.0928931, -9.7773056, -5.1688352, -3.9276476, 4.0121670
4: -11.1105347, -7.0972672, -11.0643749, -7.1003313, -3.6866026, 3.6411638
5: -0.2690043, 3.2017021, -0.2296977, 3.1932478, -3.1661773, 3.1213703
6: 4.4295435, 7.5230298, 4.4758945, 7.5128279, -2.9821663, 2.9444590
7: -18.0528088, -14.2622051, -18.0391121, -14.3019695, -2.9742146, 2.9986429
8: 0.0340765, 4.0976629, 0.0928255, 4.0821857, -3.7042723, 3.6776042
9: -8.9420567, -5.7165141, -8.8808298, -5.7248049, -2.6697536, 2.6268349

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5791
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5791

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2037729
time: 6.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2041140
time: 5.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.5993004, -5.6628609, -9.5452251, -5.7085853, -3.3152161, 3.2878757
1: -13.3701620, -8.7025909, -13.2110529, -8.7826385, -3.5942316, 3.5749779
2: -8.1649666, -4.1991200, -8.1304750, -4.3364697, -3.7496076, 3.8205881
3: -9.8313007, -5.0095692, -9.8012314, -5.1651344, -3.9752226, 4.0714965
4: -11.1851311, -7.0572872, -11.0695210, -7.0786901, -3.7478657, 3.6925159
5: -0.3350356, 3.3008976, -0.2624421, 3.1949329, -3.2332416, 3.1997750
6: 4.3850894, 7.5620174, 4.4642906, 7.5148106, -3.0357189, 3.0020232
7: -18.0840263, -14.2423973, -18.0410824, -14.2939949, -3.0128994, 3.0217524
8: -0.0389822, 4.1336994, 0.0875133, 4.0992851, -3.7655554, 3.7242966
9: -8.9749126, -5.6408854, -8.9012175, -5.7181382, -2.7107439, 2.6991901

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 5791

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 877

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029445, upper bound: 1.2022539
time: 5.61 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2025114, upper bound: 1.2025104
time: 9.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.79 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037732, upper bound: 1.2025671
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037732, upper bound: 1.2025663
IS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2022516, upper bound: 1.2017550
IS_A1_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2025110, upper bound: 1.2013198
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2029087
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2032531
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2029446, upper bound: 1.2013949
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2025114, upper bound: 1.2016547
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2034491
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037754, upper bound: 1.2037879
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2029446, upper bound: 1.2019359
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2025114, upper bound: 1.2021923
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2037729
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2037733, upper bound: 1.2041140
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2029445, upper bound: 1.2022539
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 29.79
Output dim: 6, lower bound: -1.2025114, upper bound: 1.2025104

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.5226316, -5.7150159, -9.5121460, -5.7202740, -3.0903606, 3.0869884
1: -13.1743574, -8.8530254, -13.1593304, -8.8588457, -3.4354544, 3.4245610
2: -8.0792198, -4.3618512, -8.0749216, -4.3669958, -3.6317649, 3.6317463
3: -9.7639942, -5.1820221, -9.7520266, -5.1883140, -3.8703270, 3.8646908
4: -11.0461769, -7.1258626, -11.0324783, -7.1358566, -3.5020847, 3.5048327
5: -0.2055492, 3.1809816, -0.1980662, 3.1763518, -3.0792780, 3.0763111
6: 4.4986849, 7.5008106, 4.5060940, 7.4950647, -2.8981619, 2.8955674
7: -18.0182762, -14.3364239, -18.0015373, -14.3497047, -2.9132261, 2.9109378
8: 0.1051389, 4.0556774, 0.1151752, 4.0504160, -3.5512543, 3.5408688
9: -8.8629971, -5.7595372, -8.8561745, -5.7646809, -2.5642567, 2.5645995

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2026108, upper bound: 1.2025638
time: 5.04 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037736, upper bound: 1.2025650
time: 5.34 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -9.5437288, -5.6842065, -9.5121460, -5.7202740, -3.1120043, 3.1189594
1: -13.2867527, -8.7957668, -13.1593304, -8.8588457, -3.4769554, 3.4631557
2: -8.1050110, -4.3414888, -8.0749216, -4.3669958, -3.6653275, 3.6601124
3: -9.8036795, -5.1132011, -9.7520266, -5.1883140, -3.9106846, 3.9194961
4: -11.1103592, -7.0874939, -11.0324783, -7.1358566, -3.5555820, 3.5454102
5: -0.2655878, 3.2678952, -0.1980662, 3.1763518, -3.1363764, 3.1129613
6: 4.4608111, 7.5381393, 4.5060940, 7.4950647, -2.9417372, 2.9337554
7: -18.0468807, -14.3189077, -18.0015373, -14.3497047, -2.9392672, 2.9313965
8: 0.0440102, 4.0900731, 0.1151752, 4.0504160, -3.5906868, 3.5749807
9: -8.8947382, -5.6972437, -8.8561745, -5.7646809, -2.5962954, 2.6071441

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 508
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 508

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037737, upper bound: 1.2013982
time: 5.41 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037759, upper bound: 1.2025676
time: 5.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5262566, -5.7125430, -9.5277700, -5.7122145, -3.1037669, 3.1021872
1: -13.1792183, -8.8376999, -13.1836405, -8.8365078, -3.4448280, 3.4667130
2: -8.0825396, -4.3564649, -8.0888786, -4.3541436, -3.6468725, 3.6516757
3: -9.7670155, -5.1782532, -9.7675991, -5.1768222, -3.8848238, 3.8846626
4: -11.0535526, -7.1164680, -11.0548468, -7.1147828, -3.5257263, 3.5334492
5: -0.2132001, 3.1832902, -0.2144103, 3.1848271, -3.0963221, 3.0946379
6: 4.4864039, 7.5017223, 4.4851780, 7.5034604, -2.9190130, 2.9137745
7: -18.0223541, -14.3129578, -18.0250549, -14.3114948, -2.9438753, 2.9580469
8: 0.1005213, 4.0588446, 0.0994817, 4.0618935, -3.5654020, 3.5642796
9: -8.8675709, -5.7491775, -8.8687468, -5.7454267, -2.5899982, 2.5889490

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2029094
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2029097
time: 5.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5262566, -5.7125430, -9.5489016, -5.6813993, -3.1357470, 3.1239028
1: -13.1792183, -8.8376999, -13.2960329, -8.7792511, -3.4843693, 3.4977388
2: -8.0825396, -4.3564649, -8.1147308, -4.3337741, -3.6752357, 3.6853547
3: -9.7670155, -5.1782532, -9.8072929, -5.1079893, -3.9361386, 3.9250584
4: -11.0535526, -7.1164680, -11.1190319, -7.0764213, -3.5662928, 3.5780301
5: -0.2132001, 3.1832902, -0.2744222, 3.2717481, -3.1289296, 3.1489418
6: 4.4864039, 7.5017223, 4.4472814, 7.5407858, -2.9545827, 2.9573512
7: -18.0223541, -14.3129578, -18.0536556, -14.2939844, -2.9643250, 2.9756155
8: 0.1005213, 4.0588446, 0.0383373, 4.0963225, -3.5942783, 3.5988832
9: -8.8675709, -5.7491775, -8.9004955, -5.6831217, -2.6294546, 2.6185198

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2032532
time: 5.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2032529
time: 5.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5704851, -5.7005997, -9.5203295, -5.7186537, -3.2617698, 3.2290506
1: -13.2369518, -8.7778149, -13.1813478, -8.8529091, -3.4945617, 3.4814208
2: -8.1317167, -4.2296829, -8.1063509, -4.3555465, -3.6759796, 3.7477484
3: -9.7870464, -5.0966678, -9.7550125, -5.1813612, -3.9114256, 3.9902124
4: -11.1032467, -7.1066570, -11.0393028, -7.1274137, -3.6518602, 3.6118393
5: -0.2613506, 3.1994061, -0.2042375, 3.1840966, -3.1509466, 3.0934401
6: 4.4419384, 7.5221190, 4.5000248, 7.5038481, -2.9607124, 2.9220171
7: -18.0487766, -14.2856579, -18.0150394, -14.3425274, -2.9397602, 2.9630008
8: 0.0387177, 4.0944805, 0.1100136, 4.0656996, -3.6905856, 3.6506987
9: -8.9374409, -5.7268963, -8.8622589, -5.7460675, -2.6471119, 2.5957904

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2026071, upper bound: 1.2034457
time: 5.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2034468
time: 5.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5704851, -5.7005997, -9.5418425, -5.6878281, -3.2944832, 3.2514853
1: -13.2369518, -8.7778149, -13.2937355, -8.7955761, -3.5228271, 3.5040653
2: -8.1317167, -4.2296829, -8.1321793, -4.3350997, -3.7044210, 3.7769132
3: -9.7870464, -5.0966678, -9.7947140, -5.1125054, -3.9601431, 4.0087519
4: -11.1032467, -7.1066570, -11.1038351, -7.0890265, -3.6852455, 3.6644239
5: -0.2613506, 3.1994061, -0.2643018, 3.2710400, -3.1685219, 3.1514559
6: 4.4419384, 7.5221190, 4.4620943, 7.5411711, -2.9820828, 2.9656043
7: -18.0487766, -14.2856579, -18.0436573, -14.3249216, -2.9601631, 2.9723883
8: 0.0387177, 4.0944805, 0.0488772, 4.1001687, -3.7108073, 3.6893249
9: -8.9374409, -5.7268963, -8.8941193, -5.6837397, -2.6710272, 2.6275935

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2026092, upper bound: 1.2037854
time: 5.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2037865
time: 5.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5741138, -5.6981201, -9.5367374, -5.7105856, -3.2795410, 3.2413878
1: -13.2418213, -8.7624788, -13.2056217, -8.8303900, -3.5063314, 3.5152087
2: -8.1350317, -4.2243099, -8.1202946, -4.3425598, -3.6910501, 3.7665050
3: -9.7900734, -5.0928931, -9.7706699, -5.1698804, -3.9260845, 4.0063844
4: -11.1105347, -7.0972672, -11.0628719, -7.1063442, -3.6799059, 3.6386800
5: -0.2690043, 3.2017021, -0.2206199, 3.1927631, -3.1656570, 3.1122389
6: 4.4295435, 7.5230298, 4.4791012, 7.5122614, -2.9816246, 2.9401221
7: -18.0528088, -14.2622051, -18.0385475, -14.3041697, -2.9712925, 2.9982200
8: 0.0340765, 4.0976629, 0.0943213, 4.0774384, -3.6991196, 3.6746798
9: -8.9420567, -5.7165141, -8.8751650, -5.7268009, -2.6676908, 2.6207821

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2026092, upper bound: 1.2037720
time: 5.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2037758
time: 5.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.5741138, -5.6981201, -9.5582743, -5.6797600, -3.3122730, 3.2638512
1: -13.2418213, -8.7624788, -13.3180294, -8.7730846, -3.5326753, 3.5379059
2: -8.1350317, -4.2243099, -8.1461573, -4.3221321, -3.7194967, 3.7956767
3: -9.7900734, -5.0928931, -9.8103828, -5.1010237, -3.9704003, 4.0248647
4: -11.1105347, -7.0972672, -11.1274462, -7.0679874, -3.7079935, 3.6851330
5: -0.2690043, 3.2017021, -0.2806172, 3.2797446, -3.1821938, 3.1671903
6: 4.4295435, 7.5230298, 4.4412155, 7.5495934, -2.9975300, 2.9836774
7: -18.0528088, -14.2622051, -18.0671616, -14.2866116, -2.9916563, 3.0076194
8: 0.0340765, 4.0976629, 0.0331593, 4.1119437, -3.7193680, 3.7072849
9: -8.9420567, -5.7165141, -8.9070234, -5.6644659, -2.6915994, 2.6496277

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 508
type: A, layer: 1, pos: 508
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 508

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2026073, upper bound: 1.2041119
time: 4.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2041157
time: 5.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.19 seconds
IS_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2026108, upper bound: 1.2025638
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037736, upper bound: 1.2025650
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037737, upper bound: 1.2013982
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037759, upper bound: 1.2025676
IS_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2029094
IS_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2029097
IS_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2032532
IS_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2029092, upper bound: 1.2032529
IS_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2026071, upper bound: 1.2034457
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2034468
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2026092, upper bound: 1.2037854
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2037865
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2026092, upper bound: 1.2037720
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2037758
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2026073, upper bound: 1.2041119
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.19
Output dim: 6, lower bound: -1.2037721, upper bound: 1.2041157

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.5226288, -5.7150192, -9.5121403, -5.7202768, -3.0928378, 3.0852594
1: -13.1743441, -8.8530273, -13.1593208, -8.8588505, -3.4163637, 3.3883109
2: -8.0792122, -4.3618584, -8.0749111, -4.3670044, -3.6309781, 3.6196594
3: -9.7639894, -5.1820469, -9.7520227, -5.1883440, -3.8243589, 3.8646650
4: -11.0461702, -7.1258898, -11.0324745, -7.1358867, -3.4110937, 3.4847565
5: -0.2055442, 3.1809750, -0.1980608, 3.1763434, -3.0620899, 3.0762963
6: 4.4986935, 7.5008020, 4.5061030, 7.4950533, -2.8843236, 2.9155645
7: -18.0182705, -14.3364258, -18.0015297, -14.3497076, -2.9113235, 2.9072151
8: 0.1051537, 4.0556741, 0.1151917, 4.0504122, -3.5398164, 3.5042543
9: -8.8629942, -5.7595673, -8.8561707, -5.7647138, -2.4586258, 2.5265727

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 508

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029094, upper bound: 1.2025655
time: 5.25 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029094, upper bound: 1.2025656
time: 5.34 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -9.5211420, -5.7013512, -9.5019398, -5.7263260, -3.0862389, 3.0951309
1: -13.2270947, -8.8437767, -13.1236372, -8.8646202, -3.3869057, 3.3618963
2: -8.0767422, -4.3809681, -8.0598240, -4.3809123, -3.5954342, 3.5801544
3: -9.7519321, -5.1753111, -9.7434206, -5.2309198, -3.8130713, 3.8440390
4: -11.0248871, -7.1604013, -11.0180531, -7.1919212, -3.2880454, 3.3479800
5: -0.2239647, 3.2478964, -0.1886363, 3.1634765, -3.0965281, 3.0990875
6: 4.5074224, 7.5119791, 4.5201697, 7.4771261, -2.5412211, 2.5664408
7: -18.0301704, -14.3439083, -17.9922695, -14.3573532, -2.9160910, 2.9004793
8: 0.0922966, 4.0426655, 0.1444638, 4.0419669, -3.5323477, 3.4944711
9: -8.8130398, -5.7643652, -8.8487148, -5.8194542, -2.4113503, 2.4906707

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 508

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032523, upper bound: 1.2013975
time: 4.94 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032523, upper bound: 1.2013978
time: 5.28 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -9.5437250, -5.6842103, -9.5121422, -5.7202759, -3.1102753, 3.1214399
1: -13.2867384, -8.7957706, -13.1593199, -8.8588495, -3.4260592, 3.4255996
2: -8.1050024, -4.3414955, -8.0749130, -4.3670034, -3.6531973, 3.6592865
3: -9.8036757, -5.1132288, -9.7520218, -5.1883373, -3.8921394, 3.8641191
4: -11.1103506, -7.0875220, -11.0324726, -7.1358824, -3.4984169, 3.4544206
5: -0.2655835, 3.2678866, -0.1980615, 3.1763451, -3.1272559, 3.0950737
6: 4.4608188, 7.5381279, 4.5061016, 7.4950552, -2.9576979, 2.9093699
7: -18.0468731, -14.3189049, -18.0015335, -14.3497066, -2.9354496, 2.9295173
8: 0.0440264, 4.0900674, 0.1151888, 4.0504122, -3.5479059, 3.5503502
9: -8.8947334, -5.6972771, -8.8561697, -5.7647085, -2.5401220, 2.4971294

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 508

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4597

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032523, upper bound: 1.2025642
time: 5.21 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032523, upper bound: 1.2025644
time: 5.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5704651, -5.7006292, -9.5203228, -5.7186589, -3.2498488, 3.2247820
1: -13.2368059, -8.7778397, -13.1813269, -8.8529148, -3.4648180, 3.4296117
2: -8.1316938, -4.2299113, -8.1063404, -4.3555565, -3.6759443, 3.7333093
3: -9.7870035, -5.0971255, -9.7550097, -5.1813898, -3.8664379, 3.9570374
4: -11.1031837, -7.1067066, -11.0392942, -7.1274433, -3.5790854, 3.5872712
5: -0.2613022, 3.1993837, -0.2042325, 3.1840875, -3.1339164, 3.0923676
6: 4.4420900, 7.5221076, 4.5000324, 7.5038357, -2.9456587, 2.9417105
7: -18.0487328, -14.2856722, -18.0150318, -14.3425322, -2.9397135, 2.9553747
8: 0.0387708, 4.0944109, 0.1100307, 4.0656939, -3.6654816, 3.6225467
9: -8.9374352, -5.7270031, -8.8622541, -5.7461033, -2.5423598, 2.5565815

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4597
type: A, layer: 1, pos: 508

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 946

## Relational analysis of IS_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 555

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2033799, upper bound: 1.2034464
time: 9.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037735, upper bound: 1.2034464
time: 5.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.5589056, -5.7072115, -9.5185108, -5.7049875, -3.1630998, 3.1054430
1: -13.1996365, -8.7841549, -13.2340899, -8.8438473, -3.4534869, 3.4224107
2: -8.1166105, -4.2461891, -8.1038694, -4.3751168, -3.6269279, 3.7299707
3: -9.7779236, -5.1437144, -9.7429218, -5.1749845, -3.8835077, 3.9043303
4: -11.0854454, -7.1628995, -11.0173168, -7.1619425, -3.5129566, 3.3142481
5: -0.2513440, 3.1858552, -0.2222638, 3.2509274, -3.1574864, 3.0987422
6: 4.4583578, 7.5042620, 4.5088215, 7.5151033, -2.9412904, 2.5645761
7: -18.0391293, -14.2939787, -18.0269871, -14.3504105, -2.9371099, 2.9538531
8: 0.0682049, 4.0851479, 0.0972959, 4.0526991, -3.5533700, 3.5532289
9: -8.9292984, -5.7825985, -8.8121243, -5.7508163, -2.5457697, 2.4682829

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 508
type: B, layer: 1, pos: 4597

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 946

## Relational analysis of IS_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.956814765930176
rel_dist={6: [-1.2041204236983045, 1.204119556210955]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 2428.39 seconds
