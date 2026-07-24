## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.64315406431
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.6137395, 2.6137395)
1: (-11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.9504948, 2.9504948)
2: (-10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.2090549, 2.2090549)
3: (-4.4308734, -2.2690167, -4.4308734, -2.2690167, -2.1618567, 2.1618567)
4: (-15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.6839581, 2.6839581)
5: (8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.5050173, 1.5050173)
6: (-4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.4684534, 2.4684534)
7: (-15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.8441019, 2.8441019)
8: (-0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.7412529, 1.7412528)
9: (-6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154)

## BASE Result
execution time: IAR + LP analysis = 15.15 + 32.02 = 47.17 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.83 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.2053476572036743
rel_dist={5: [-0.865066048868588, 0.8650657311232219]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.0063066482543945
rel_dist={5: [-0.6116870135107089, 0.6116892874758619]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.0726536512374878
rel_dist={5: [-0.7059536409330516, 0.7059557061852892]}

## Binary Search Result
Binary search time: 151.93 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual_ind) starts
Time budget: 3400.90 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9828011
time: 3.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687710
time: 4.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.28
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9828011
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.28
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687710

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.1630764, -10.4619446, -2.1303844, 2.0840201
1: -11.2995148, -8.4136467, -11.3005276, -8.3500328, -2.4721837, 2.4288750
2: -10.7255239, -8.5416927, -10.7255383, -8.5164833, -2.1620536, 2.1347435
3: -4.4290442, -2.3468359, -4.4308734, -2.2690167, -1.9831219, 1.9484730
4: -15.1385698, -12.5005589, -15.1825619, -12.4986038, -2.3023939, 2.3203721
5: 8.2217073, 9.7064924, 8.2191973, 9.7242146, -1.3316764, 1.3099520
6: -4.7417126, -2.3063393, -4.7438617, -2.2754083, -2.0770645, 2.0488894
7: -15.7444887, -12.9351664, -15.7447653, -12.9006634, -2.6720085, 2.6286469
8: -0.8229923, 0.9184313, -0.8548889, 0.9188228, -1.3952689, 1.4549845
9: -6.6816912, -5.0027695, -6.7094598, -5.0025444, -1.6791468, 1.7066903

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687699
time: 4.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687705
time: 4.24 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3069639, -10.4817972, -13.1623383, -10.4659634, -2.2437539, 2.1864734
1: -11.4256668, -8.4219580, -11.3000221, -8.3601360, -2.6449308, 2.4944756
2: -10.7750311, -8.5532455, -10.7255230, -8.5212917, -2.2396545, 2.1445441
3: -4.5807705, -2.3410683, -4.4302597, -2.2773366, -2.0178790, 1.9974186
4: -15.1519251, -12.4215012, -15.1786699, -12.4991131, -2.3223801, 2.2396767
5: 8.2167139, 9.6680660, 8.2200565, 9.7179613, -1.3862062, 1.2892220
6: -4.7917042, -2.3123016, -4.7428989, -2.2798810, -2.0818777, 2.0834899
7: -15.7892628, -12.9812613, -15.7446060, -12.9107704, -2.7600489, 2.3642197
8: -0.7966328, 0.9715328, -0.8461232, 0.9185996, -1.3754590, 1.5877769
9: -6.6794314, -4.9480395, -6.7055421, -5.0026388, -1.5287342, 1.7575026

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687718
time: 3.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687719
time: 3.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.77 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 12.77
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687699
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.77
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687705
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.77
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687718
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.77
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687719

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.1613884, -10.5266781, -2.0827703, 2.0827701
1: -11.2995148, -8.4136467, -11.2995148, -8.4136467, -2.4278975, 2.4278972
2: -10.7255239, -8.5416927, -10.7255239, -8.5416927, -2.1337028, 2.1337023
3: -4.4290442, -2.3468359, -4.4290442, -2.3468359, -1.9478369, 1.9478369
4: -15.1385698, -12.5005589, -15.1385698, -12.5005589, -2.3008122, 2.3008125
5: 8.2217073, 9.7064924, 8.2217073, 9.7064924, -1.3035867, 1.3035867
6: -4.7417126, -2.3063393, -4.7417126, -2.3063393, -2.0471091, 2.0471091
7: -15.7444887, -12.9351664, -15.7444887, -12.9351664, -2.6206322, 2.6206317
8: -0.8229923, 0.9184313, -0.8229923, 0.9184313, -1.3944390, 1.3944392
9: -6.6816912, -5.0027695, -6.6816912, -5.0027695, -1.6789217, 1.6789217

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9718186
time: 4.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9718645
time: 4.94 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.3069639, -10.4817972, -2.1101365, 2.1869528
1: -11.2995148, -8.4136467, -11.4256668, -8.4219580, -2.4511909, 2.6064987
2: -10.7255239, -8.5416927, -10.7750311, -8.5532455, -2.1373830, 2.2158239
3: -4.4290442, -2.3468359, -4.5807705, -2.3410683, -1.9457092, 1.9721990
4: -15.1385698, -12.5005589, -15.1519251, -12.4215012, -2.2160702, 2.2932951
5: 8.2217073, 9.7064924, 8.2167139, 9.6680660, -1.3092391, 1.3600233
6: -4.7417126, -2.3063393, -4.7917042, -2.3123016, -2.0442338, 2.0495405
7: -15.7444887, -12.9351664, -15.7892628, -12.9812613, -2.3477221, 2.7152262
8: -0.8229923, 0.9184313, -0.7966328, 0.9715328, -1.5281777, 1.3928515
9: -6.6816912, -5.0027695, -6.6794314, -4.9480395, -1.7336516, 1.4862633

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9718202
time: 3.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9718661
time: 3.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.3069639, -10.4817972, -13.1613884, -10.5266781, -2.1869521, 2.1101365
1: -11.4256668, -8.4219580, -11.2995148, -8.4136467, -2.6064992, 2.4511902
2: -10.7750311, -8.5532455, -10.7255239, -8.5416927, -2.2158237, 2.1373827
3: -4.5807705, -2.3410683, -4.4290442, -2.3468359, -1.9721985, 1.9457097
4: -15.1519251, -12.4215012, -15.1385698, -12.5005589, -2.2932949, 2.2160702
5: 8.2167139, 9.6680660, 8.2217073, 9.7064924, -1.3600235, 1.3092391
6: -4.7917042, -2.3123016, -4.7417126, -2.3063393, -2.0495405, 2.0442338
7: -15.7892628, -12.9812613, -15.7444887, -12.9351664, -2.7152262, 2.3477223
8: -0.7966328, 0.9715328, -0.8229923, 0.9184313, -1.3928516, 1.5281780
9: -6.6794314, -4.9480395, -6.6816912, -5.0027695, -1.4862633, 1.7336516

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577891
time: 5.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369
time: 3.71 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3069639, -10.4817972, -13.3069639, -10.4817972, -2.1432152, 2.1432152
1: -11.4256668, -8.4219580, -11.4256668, -8.4219580, -2.4913645, 2.4913642
2: -10.7750311, -8.5532455, -10.7750311, -8.5532455, -2.1379066, 2.1379068
3: -4.5807705, -2.3410683, -4.5807705, -2.3410683, -1.8086405, 1.8086405
4: -15.1519251, -12.4215012, -15.1519251, -12.4215012, -2.1579351, 2.1579351
5: 8.2167139, 9.6680660, 8.2167139, 9.6680660, -1.2758293, 1.2758293
6: -4.7917042, -2.3123016, -4.7917042, -2.3123016, -1.9761369, 1.9761369
7: -15.7892628, -12.9812613, -15.7892628, -12.9812613, -2.3659678, 2.3659680
8: -0.7966328, 0.9715328, -0.7966328, 0.9715328, -1.3710752, 1.3710752
9: -6.6794314, -4.9480395, -6.6794314, -4.9480395, -1.5284581, 1.5284584

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577911
time: 3.86 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369
time: 4.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.86 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9718186
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9718645
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9718202
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9718661
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577891
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577911
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1633673, -10.5471478, -13.1613808, -10.5297728, -2.0766058, 2.0626311
1: -11.3239450, -8.4196949, -11.2995062, -8.4145632, -2.4439116, 2.4229288
2: -10.7499352, -8.5666199, -10.7255230, -8.5475016, -2.1046722, 2.0823686
3: -4.3989763, -2.3425825, -4.4236903, -2.3468392, -1.9115129, 1.9248927
4: -15.1122828, -12.5838108, -15.1385679, -12.5178137, -2.2662220, 2.2252440
5: 8.2515154, 9.7071228, 8.2293205, 9.7064915, -1.2665992, 1.2920458
6: -4.7121363, -2.3152876, -4.7367353, -2.3063402, -2.0144968, 2.0276051
7: -15.7508335, -12.9453793, -15.7444878, -12.9369030, -2.6282291, 2.6073012
8: -0.7826340, 0.9031253, -0.8168800, 0.9184299, -1.3504949, 1.3700259
9: -6.6767054, -5.0664907, -6.6816845, -5.0127063, -1.6639991, 1.6151938

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674947
time: 3.55 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9496548, upper bound: 0.9680572
time: 4.05 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1613884, -10.5266781, -2.0827446, 2.0654981
1: -11.2994614, -8.4265642, -11.2995148, -8.4136467, -2.4278517, 2.4246356
2: -10.7255049, -8.5611706, -10.7255239, -8.5416927, -2.1336284, 2.0590923
3: -4.4181614, -2.3468671, -4.4290442, -2.3468359, -1.9099841, 1.9477053
4: -15.1385651, -12.5070190, -15.1385698, -12.5005589, -2.3007488, 2.2425628
5: 8.2271509, 9.7064800, 8.2217073, 9.7064924, -1.2843874, 1.3035784
6: -4.7387886, -2.3063402, -4.7417126, -2.3063393, -2.0140896, 2.0470607
7: -15.7444801, -12.9415779, -15.7444887, -12.9351664, -2.6205139, 2.6292219
8: -0.8204088, 0.9184208, -0.8229923, 0.9184313, -1.3514812, 1.3944074
9: -6.6816416, -5.0141444, -6.6816912, -5.0027695, -1.6788721, 1.6675467

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624619
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624618
time: 3.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1633673, -10.5471478, -13.3069572, -10.4848957, -2.1039720, 2.1629219
1: -11.3239450, -8.4196949, -11.4256582, -8.4228706, -2.4672041, 2.6015253
2: -10.7499352, -8.5666199, -10.7750320, -8.5590572, -2.1087027, 2.1644905
3: -4.3989763, -2.3425825, -4.5754194, -2.3410733, -1.9093871, 1.9556568
4: -15.1122828, -12.5838108, -15.1519241, -12.4387512, -2.1855860, 2.2177267
5: 8.2515154, 9.7071228, 8.2243671, 9.6680660, -1.2707839, 1.3484827
6: -4.7121363, -2.3152876, -4.7867308, -2.3123019, -2.0116205, 2.0335584
7: -15.7508335, -12.9453793, -15.7892618, -12.9829006, -2.3592753, 2.7018938
8: -0.7826340, 0.9031253, -0.7905209, 0.9715302, -1.4842339, 1.3698514
9: -6.6767054, -5.0664907, -6.6794300, -4.9579768, -1.7187285, 1.4161232

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572203
time: 3.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9253513, upper bound: 0.9577829
time: 3.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.3069639, -10.4817972, -2.1101108, 2.1552191
1: -11.2994614, -8.4265642, -11.4256668, -8.4219580, -2.4511452, 2.6032529
2: -10.7255049, -8.5611706, -10.7750311, -8.5532455, -2.1373119, 2.1412148
3: -4.4181614, -2.3468671, -4.5807705, -2.3410683, -1.9079008, 1.9721503
4: -15.1385651, -12.5070190, -15.1519251, -12.4215012, -2.2160683, 2.2350657
5: 8.2271509, 9.7064800, 8.2167139, 9.6680660, -1.2840068, 1.3600149
6: -4.7387886, -2.3063402, -4.7917042, -2.3123016, -2.0112109, 2.0495396
7: -15.7444801, -12.9415779, -15.7892628, -12.9812613, -2.3477154, 2.7238812
8: -0.8204088, 0.9184208, -0.7966328, 0.9715328, -1.4852202, 1.3928378
9: -6.6816416, -5.0141444, -6.6794314, -4.9480395, -1.7336020, 1.4051397

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9521876
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9718661
time: 3.62 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3090773, -10.5022688, -13.1613808, -10.5297728, -2.1713262, 2.0899982
1: -11.4500942, -8.4280024, -11.2995062, -8.4145632, -2.6225247, 2.4462218
2: -10.7994423, -8.5782375, -10.7255230, -8.5475016, -2.1867940, 2.0867274
3: -4.5506849, -2.3368161, -4.4236903, -2.3468392, -1.9386005, 1.9228182
4: -15.1256447, -12.5048466, -15.1385679, -12.5178137, -2.2587261, 2.1441574
5: 8.2465706, 9.6687326, 8.2293205, 9.7064915, -1.3230305, 1.2937901
6: -4.7621164, -2.3212516, -4.7367353, -2.3063402, -2.0211263, 2.0247257
7: -15.7956581, -12.9913330, -15.7444878, -12.9369030, -2.7228880, 2.3403568
8: -0.7562866, 0.9562299, -0.8168800, 0.9184299, -1.3522344, 1.5037649
9: -6.6745906, -5.0117588, -6.6816845, -5.0127063, -1.4459429, 1.6699257

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
time: 3.94 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393804, upper bound: 0.9437539
time: 3.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.1613884, -10.5266781, -2.1869287, 2.0928640
1: -11.4256086, -8.4348717, -11.2995148, -8.4136467, -2.6064515, 2.4479291
2: -10.7750120, -8.5727234, -10.7255239, -8.5416927, -2.2157502, 2.0636506
3: -4.5699735, -2.3410995, -4.4290442, -2.3468359, -1.9444647, 1.9455805
4: -15.1519194, -12.4279480, -15.1385698, -12.5005589, -2.2932301, 2.1796453
5: 8.2221375, 9.6680603, 8.2217073, 9.7064924, -1.3408232, 1.3092346
6: -4.7887893, -2.3123035, -4.7417126, -2.3063393, -2.0249281, 2.0441849
7: -15.7892513, -12.9876451, -15.7444887, -12.9351664, -2.7151070, 2.3697248
8: -0.7940555, 0.9715207, -0.8229923, 0.9184313, -1.3584471, 1.5281458
9: -6.6794038, -4.9594150, -6.6816912, -5.0027695, -1.4862170, 1.7222762

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9381584
time: 3.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9578369
time: 3.66 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3090773, -10.5022688, -13.3069572, -10.4848957, -2.1275911, 2.1191847
1: -11.4500942, -8.4280024, -11.4256582, -8.4228706, -2.5073962, 2.4863918
2: -10.7994423, -8.5782375, -10.7750320, -8.5590572, -2.1092267, 2.0872509
3: -4.5506849, -2.3368161, -4.5754194, -2.3410733, -1.7750421, 1.7921679
4: -15.1256447, -12.5048466, -15.1519241, -12.4387512, -2.1275129, 2.0860250
5: 8.2465706, 9.6687326, 8.2243671, 9.6680660, -1.2373620, 1.2603657
6: -4.7621164, -2.3212516, -4.7867308, -2.3123019, -1.9477251, 1.9601834
7: -15.7956581, -12.9913330, -15.7892618, -12.9829006, -2.3775463, 2.3586013
8: -0.7562866, 0.9562299, -0.7905209, 0.9715302, -1.3304453, 1.3480916
9: -6.6745906, -5.0117588, -6.6794300, -4.9579768, -1.4881475, 1.4583268

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
time: 3.69 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9253513, upper bound: 0.9437539
time: 3.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.3069639, -10.4817972, -2.1431923, 2.1114821
1: -11.4256086, -8.4348717, -11.4256668, -8.4219580, -2.4913187, 2.4881191
2: -10.7750120, -8.5727234, -10.7750311, -8.5532455, -2.1378374, 2.0641735
3: -4.5699735, -2.3410995, -4.5807705, -2.3410683, -1.7809796, 1.8085902
4: -15.1519194, -12.4279480, -15.1519251, -12.4215012, -2.1579332, 2.1215727
5: 8.2221375, 9.6680603, 8.2167139, 9.6680660, -1.2505832, 1.2758243
6: -4.7887893, -2.3123035, -4.7917042, -2.3123016, -1.9515543, 1.9761357
7: -15.7892513, -12.9876451, -15.7892628, -12.9812613, -2.3659611, 2.3879948
8: -0.7940555, 0.9715207, -0.7966328, 0.9715328, -1.3366816, 1.3710614
9: -6.6794038, -4.9594150, -6.6794314, -4.9480395, -1.5284119, 1.4473436

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9381585
time: 3.94 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9578370
time: 3.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.18 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674947
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9496548, upper bound: 0.9680572
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624619
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624618
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572203
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9253513, upper bound: 0.9577829
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9521876
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9718661
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9393804, upper bound: 0.9437539
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9381584
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9578369
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9253513, upper bound: 0.9437539
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9381585
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.18
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9578370

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1632938, -10.5476933, -13.1566582, -10.5510864, -2.0377531, 2.0577793
1: -11.3238506, -8.4201937, -11.2935362, -8.4465971, -2.3879299, 2.4184878
2: -10.7488365, -8.5667019, -10.6638451, -8.5526924, -2.1018889, 2.0087514
3: -4.3989482, -2.3426192, -4.4218569, -2.3489740, -1.9092364, 1.9238377
4: -15.1122799, -12.5839281, -15.1383734, -12.5253992, -2.2601094, 2.2227378
5: 8.2515297, 9.7067852, 8.2301731, 9.6845121, -1.2451425, 1.2865032
6: -4.7121048, -2.3158014, -4.7347169, -2.3376522, -1.9726992, 2.0251532
7: -15.7507696, -12.9455805, -15.7401857, -12.9499035, -2.6166191, 2.6031723
8: -0.7822597, 0.9031215, -0.7926233, 0.9181724, -1.3466382, 1.3449076
9: -6.6765742, -5.0665269, -6.6729708, -5.0150967, -1.6614776, 1.6064439

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612831
time: 3.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674927
time: 5.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1631575, -10.5481625, -13.1864090, -10.5393200, -2.0494385, 2.1138482
1: -11.3237076, -8.4252548, -11.3032475, -8.4618835, -2.3810868, 2.4915321
2: -10.7469082, -8.5667763, -10.7027826, -8.4952068, -2.1943474, 2.0307870
3: -4.3989224, -2.3427069, -4.4281187, -2.3455315, -1.9119239, 1.9302835
4: -15.1122723, -12.5854073, -15.1351528, -12.5331144, -2.2609076, 2.2176988
5: 8.2515602, 9.7052383, 8.2176399, 9.6887217, -1.2579033, 1.2918013
6: -4.7120304, -2.3168576, -4.7648597, -2.3143473, -1.9904041, 2.0746751
7: -15.7507153, -12.9467287, -15.7503281, -12.9482136, -2.6154861, 2.6061110
8: -0.7812953, 0.9031134, -0.8115044, 0.9446242, -1.3652439, 1.3780036
9: -6.6755514, -5.0665808, -6.6713901, -5.0119200, -1.6636314, 1.6048093

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9612814
time: 5.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9680573
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1633673, -10.5471478, -2.0626106, 2.0757682
1: -11.2994614, -8.4265642, -11.3239450, -8.4196949, -2.4228878, 2.4314227
2: -10.7255049, -8.5611706, -10.7499352, -8.5666199, -2.0822997, 2.1116252
3: -4.4181614, -2.3468671, -4.3989763, -2.3425825, -1.9278574, 1.9113874
4: -15.1385651, -12.5070190, -15.1122828, -12.5838108, -2.2251816, 2.2746201
5: 8.2271509, 9.7064800, 8.2515154, 9.7071228, -1.2921325, 1.2665920
6: -4.7387886, -2.3063402, -4.7121363, -2.3152876, -2.0326285, 2.0144491
7: -15.7444801, -12.9415779, -15.7508335, -12.9453793, -2.6071854, 2.6193652
8: -0.8204088, 0.9184208, -0.7826340, 0.9031253, -1.3763700, 1.3504651
9: -6.6816416, -5.0141444, -6.6767054, -5.0664907, -1.6151509, 1.6625609

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9674926, upper bound: 0.9416506
time: 3.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9680553, upper bound: 0.9484246
time: 3.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1613560, -10.5350914, -2.0654745, 2.0654747
1: -11.2994614, -8.4265642, -11.2994614, -8.4265642, -2.4245901, 2.4245899
2: -10.7255049, -8.5611706, -10.7255049, -8.5611706, -2.0590198, 2.0590196
3: -4.4181614, -2.3468671, -4.4181614, -2.3468671, -1.9099169, 1.9099166
4: -15.1385651, -12.5070190, -15.1385651, -12.5070190, -2.2425370, 2.2425373
5: 8.2271509, 9.7064800, 8.2271509, 9.7064800, -1.2843792, 1.2843794
6: -4.7387886, -2.3063402, -4.7387886, -2.3063402, -2.0140619, 2.0140619
7: -15.7444801, -12.9415779, -15.7444801, -12.9415779, -2.6291733, 2.6291728
8: -0.8204088, 0.9184208, -0.8204088, 0.9184208, -1.3514493, 1.3514493
9: -6.6816416, -5.0141444, -6.6816416, -5.0141444, -1.6674972, 1.6674972

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9674928, upper bound: 0.9421956
time: 3.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9680555, upper bound: 0.9489707
time: 3.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1632938, -10.5476933, -13.3020706, -10.5062084, -2.0651188, 2.1578431
1: -11.3238506, -8.4201937, -11.4193640, -8.4549103, -2.4112248, 2.5963175
2: -10.7488365, -8.5667019, -10.7133293, -8.5639191, -2.1054835, 2.0908914
3: -4.3989482, -2.3426192, -4.5735259, -2.3432255, -1.9068379, 1.9542542
4: -15.1122799, -12.5839281, -15.1517277, -12.4463320, -2.1751766, 2.2152715
5: 8.2515297, 9.7067852, 8.2252398, 9.6461134, -1.2470052, 1.3429466
6: -4.7121048, -2.3158014, -4.7849193, -2.3436356, -1.9698176, 2.0313885
7: -15.7507696, -12.9455805, -15.7843103, -12.9958458, -2.3462048, 2.6966982
8: -0.7822597, 0.9031215, -0.7661648, 0.9712572, -1.4803820, 1.3426992
9: -6.6765742, -5.0665269, -6.6707277, -4.9604282, -1.7161460, 1.4014974

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9510089
time: 3.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572204
time: 3.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1631575, -10.5481625, -13.3320189, -10.4944439, -2.0768042, 2.2091179
1: -11.3237076, -8.4252548, -11.4292145, -8.4701881, -2.4043808, 2.6699591
2: -10.7469082, -8.5667763, -10.7522383, -8.5062780, -2.1982498, 2.1129360
3: -4.3989224, -2.3427069, -4.5798178, -2.3397806, -1.9094148, 1.9595957
4: -15.1122723, -12.5854073, -15.1485243, -12.4540434, -2.1752176, 2.2105229
5: 8.2515602, 9.7052383, 8.2126656, 9.6503067, -1.2591794, 1.3491694
6: -4.7120304, -2.3168576, -4.8156328, -2.3203645, -1.9875045, 2.0767906
7: -15.7507153, -12.9467287, -15.7946587, -12.9941196, -2.3506155, 2.6999602
8: -0.7812953, 0.9031134, -0.7849360, 0.9978013, -1.4991646, 1.3742805
9: -6.6755514, -5.0665808, -6.6690588, -4.9572344, -1.7183170, 1.4044998

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9510069
time: 5.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9577820
time: 4.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.3090773, -10.5022688, -2.0899777, 2.1742637
1: -11.2994614, -8.4265642, -11.4500942, -8.4280024, -2.4461803, 2.6100357
2: -10.7255049, -8.5611706, -10.7994423, -8.5782375, -2.0866618, 2.1937470
3: -4.4181614, -2.3468671, -4.5506849, -2.3368161, -1.9257832, 1.9385562
4: -15.1385651, -12.5070190, -15.1256447, -12.5048466, -2.1441565, 2.2671242
5: 8.2271509, 9.7064800, 8.2465706, 9.6687326, -1.2974241, 1.3230233
6: -4.7387886, -2.3063402, -4.7621164, -2.3212516, -2.0297494, 2.0211263
7: -15.7444801, -12.9415779, -15.7956581, -12.9913330, -2.3403506, 2.7140236
8: -0.8204088, 0.9184208, -0.7562866, 0.9562299, -1.5101089, 1.3522218
9: -6.6816416, -5.0141444, -6.6745906, -5.0117588, -1.6698828, 1.4591260

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9313763
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9437519, upper bound: 0.9381503
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.3069401, -10.4902096, -2.0928402, 2.1551957
1: -11.2994614, -8.4265642, -11.4256086, -8.4348717, -2.4478827, 2.6032057
2: -10.7255049, -8.5611706, -10.7750120, -8.5727234, -2.0635810, 2.1411417
3: -4.4181614, -2.3468671, -4.5699735, -2.3410995, -1.9078369, 1.9444265
4: -15.1385651, -12.5070190, -15.1519194, -12.4279480, -2.1796427, 2.2350407
5: 8.2271509, 9.7064800, 8.2221375, 9.6680603, -1.2840025, 1.3408149
6: -4.7387886, -2.3063402, -4.7887893, -2.3123035, -2.0111833, 2.0249271
7: -15.7444801, -12.9415779, -15.7892513, -12.9876451, -2.3697186, 2.7238302
8: -0.8204088, 0.9184208, -0.7940555, 0.9715207, -1.4851882, 1.3584330
9: -6.6816416, -5.0141444, -6.6794038, -4.9594150, -1.7222266, 1.4051039

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9319222
time: 3.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9386964
time: 4.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.3090010, -10.5028172, -13.1566582, -10.5510864, -2.1364980, 2.0851469
1: -11.4499931, -8.4285011, -11.2935362, -8.4465971, -2.5665402, 2.4417806
2: -10.7983446, -8.5783138, -10.6638451, -8.5526924, -2.1840112, 2.0131693
3: -4.5506554, -2.3368518, -4.4218569, -2.3489740, -1.9366703, 1.9217591
4: -15.1256409, -12.5049629, -15.1383734, -12.5253992, -2.2526150, 2.1416450
5: 8.2465858, 9.6683960, 8.2301731, 9.6845121, -1.3015728, 1.2884603
6: -4.7620850, -2.3217676, -4.7347169, -2.3376522, -1.9797475, 2.0222747
7: -15.7955828, -12.9915333, -15.7401857, -12.9499035, -2.7112598, 2.3387961
8: -0.7559106, 0.9562240, -0.7926233, 0.9181724, -1.3487902, 1.4786463
9: -6.6744580, -5.0117989, -6.6729708, -5.0150967, -1.4439082, 1.6611719

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
time: 3.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431905
time: 3.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.3088646, -10.5032864, -13.1864090, -10.5393200, -2.1477566, 2.1412151
1: -11.4498463, -8.4335632, -11.3032475, -8.4618835, -2.5596771, 2.5148249
2: -10.7964144, -8.5783825, -10.7027826, -8.4952068, -2.2764702, 2.0354395
3: -4.5506287, -2.3369343, -4.4281187, -2.3455315, -1.9398222, 1.9281893
4: -15.1256332, -12.5064383, -15.1351528, -12.5331144, -2.2534084, 2.1450338
5: 8.2466173, 9.6668530, 8.2176399, 9.6887217, -1.3143322, 1.2980503
6: -4.7620201, -2.3228233, -4.7648597, -2.3143473, -2.0016434, 2.0717943
7: -15.7955198, -12.9926767, -15.7503281, -12.9482136, -2.7101135, 2.3485003
8: -0.7549405, 0.9562154, -0.8115044, 0.9446242, -1.3710515, 1.5117426
9: -6.6734362, -5.0118523, -6.6713901, -5.0119200, -1.4665122, 1.6595378

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9369782
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9437539
time: 4.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.1633673, -10.5471478, -2.1629019, 2.1031342
1: -11.4256086, -8.4348717, -11.3239450, -8.4196949, -2.6014843, 2.4547157
2: -10.7750120, -8.5727234, -10.7499352, -8.5666199, -2.1644216, 2.1152086
3: -4.5699735, -2.3410995, -4.3989763, -2.3425825, -1.9532847, 1.9092627
4: -15.1519194, -12.4279480, -15.1122828, -12.5838108, -2.2176633, 2.1904612
5: 8.2221375, 9.6680603, 8.2515154, 9.7071228, -1.3486019, 1.2707791
6: -4.7887893, -2.3123035, -4.7121363, -2.3152876, -2.0380969, 2.0115733
7: -15.7892513, -12.9876451, -15.7508335, -12.9453793, -2.7017784, 2.3543105
8: -0.7940555, 0.9715207, -0.7826340, 0.9031253, -1.3754447, 1.4842033
9: -6.6794038, -4.9594150, -6.6767054, -5.0664907, -1.4160786, 1.7172904

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9572183, upper bound: 0.9173458
time: 5.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577810, upper bound: 0.9241211
time: 3.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.1613560, -10.5350914, -2.1551957, 2.0928404
1: -11.4256086, -8.4348717, -11.2994614, -8.4265642, -2.6032057, 2.4478829
2: -10.7750120, -8.5727234, -10.7255049, -8.5611706, -2.1411417, 2.0635810
3: -4.5699735, -2.3410995, -4.4181614, -2.3468671, -1.9444265, 1.9078362
4: -15.1519194, -12.4279480, -15.1385651, -12.5070190, -2.2350407, 2.1796427
5: 8.2221375, 9.6680603, 8.2271509, 9.7064800, -1.3408151, 1.2840025
6: -4.7887893, -2.3123035, -4.7387886, -2.3063402, -2.0249267, 2.0111833
7: -15.7892513, -12.9876451, -15.7444801, -12.9415779, -2.7238302, 2.3697178
8: -0.7940555, 0.9715207, -0.8204088, 0.9184208, -1.3584330, 1.4851882
9: -6.6794038, -4.9594150, -6.6816416, -5.0141444, -1.4051042, 1.7222266

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9572185, upper bound: 0.9178909
time: 5.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577812, upper bound: 0.9246672
time: 3.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3090010, -10.5028172, -13.3020706, -10.5062084, -2.0927653, 2.1144133
1: -11.4499931, -8.4285011, -11.4193640, -8.4549103, -2.4514241, 2.4819586
2: -10.7983446, -8.5783138, -10.7133293, -8.5639191, -2.1064672, 2.0137124
3: -4.5506554, -2.3368518, -4.5735259, -2.3432255, -1.7731409, 1.7910397
4: -15.1256409, -12.5049629, -15.1517277, -12.4463320, -2.1172938, 2.0835166
5: 8.2465858, 9.6683960, 8.2252398, 9.6461134, -1.2134784, 1.2550257
6: -4.7620850, -2.3217676, -4.7849193, -2.3436356, -1.9063523, 1.9578223
7: -15.7955828, -12.9915333, -15.7843103, -12.9958458, -2.3643837, 2.3570337
8: -0.7559106, 0.9562240, -0.7661648, 0.9712572, -1.3270874, 1.3214310
9: -6.6744580, -5.0117989, -6.6707277, -4.9604282, -1.4861500, 1.4437015

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9369801
time: 3.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
time: 3.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3088646, -10.5032864, -13.3320189, -10.4944439, -2.1040342, 2.1658301
1: -11.4498463, -8.4335632, -11.4292145, -8.4701881, -2.4445686, 2.5553730
2: -10.7964144, -8.5783825, -10.7522383, -8.5062780, -2.1990628, 2.0360007
3: -4.5506287, -2.3369343, -4.5798178, -2.3397806, -1.7763047, 1.7962813
4: -15.1256332, -12.5064383, -15.1485243, -12.4540434, -2.1172566, 2.0872641
5: 8.2466173, 9.6668530, 8.2126656, 9.6503067, -1.2256746, 1.2655518
6: -4.7620201, -2.3228233, -4.8156328, -2.3203645, -1.9282343, 2.0031519
7: -15.7955198, -12.9926767, -15.7946587, -12.9941196, -2.3686810, 2.3667583
8: -0.7549405, 0.9562154, -0.7849360, 0.9978013, -1.3495069, 1.3532706
9: -6.6734362, -5.0118523, -6.6690588, -4.9572344, -1.5087717, 1.4467053

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9369801
time: 3.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9437539
time: 3.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.3090773, -10.5022688, -2.1191659, 2.1305296
1: -11.4256086, -8.4348717, -11.4500942, -8.4280024, -2.4863515, 2.4949076
2: -10.7750120, -8.5727234, -10.7994423, -8.5782375, -2.0871859, 2.1157310
3: -4.5699735, -2.3410995, -4.5506849, -2.3368161, -1.7897964, 1.7749960
4: -15.1519194, -12.4279480, -15.1256447, -12.5048466, -2.0860243, 2.1323874
5: 8.2221375, 9.6680603, 8.2465706, 9.6687326, -1.2640805, 1.2373576
6: -4.7887893, -2.3123035, -4.7621164, -2.3212516, -1.9647241, 1.9477251
7: -15.7892513, -12.9876451, -15.7956581, -12.9913330, -2.3585949, 2.3725815
8: -0.7940555, 0.9715207, -0.7562866, 0.9562299, -1.3536804, 1.3304332
9: -6.6794038, -4.9594150, -6.6745906, -5.0117588, -1.4582822, 1.5013198

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9173474
time: 3.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9437519, upper bound: 0.9241212
time: 3.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.3069401, -10.4902096, -2.1114597, 2.1114600
1: -11.4256086, -8.4348717, -11.4256086, -8.4348717, -2.4880743, 2.4880741
2: -10.7750120, -8.5727234, -10.7750120, -8.5727234, -2.0641036, 2.0641038
3: -4.5699735, -2.3410995, -4.5699735, -2.3410995, -1.7809401, 1.7809399
4: -15.1519194, -12.4279480, -15.1519194, -12.4279480, -2.1215703, 2.1215703
5: 8.2221375, 9.6680603, 8.2221375, 9.6680603, -1.2505786, 1.2505786
6: -4.7887893, -2.3123035, -4.7887893, -2.3123035, -1.9515543, 1.9515541
7: -15.7892513, -12.9876451, -15.7892513, -12.9876451, -2.3879886, 2.3879886
8: -0.7940555, 0.9715207, -0.7940555, 0.9715207, -1.3366685, 1.3366684
9: -6.6794038, -4.9594150, -6.6794038, -4.9594150, -1.4473071, 1.4473071

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9178934
time: 3.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9246672
time: 3.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612831
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674927
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9612814
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9680573
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9674926, upper bound: 0.9416506
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9680553, upper bound: 0.9484246
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9674928, upper bound: 0.9421956
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9680555, upper bound: 0.9489707
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9510089
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572204
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9510069
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9577820
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9313763
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9437519, upper bound: 0.9381503
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9319222
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9386964
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431905
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9369782
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9437539
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9572183, upper bound: 0.9173458
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9577810, upper bound: 0.9241211
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9572185, upper bound: 0.9178909
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9577812, upper bound: 0.9246672
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9369801
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9369801
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9437539
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9173474
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9437519, upper bound: 0.9241212
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9178934
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.18
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9246672

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.1566582, -10.5510864, -2.0334735, 2.0196099
1: -11.3179722, -8.4517279, -11.2935362, -8.4465971, -2.3845901, 2.3635817
2: -10.6882601, -8.5720196, -10.6638451, -8.5526924, -2.0294783, 2.0068927
3: -4.3971014, -2.3447456, -4.4218569, -2.3489740, -1.9082146, 1.9215946
4: -15.1120892, -12.5913963, -15.1383734, -12.5253992, -2.2578001, 2.2168281
5: 8.2523460, 9.6852093, 8.2301731, 9.6845121, -1.2401841, 1.2656863
6: -4.7101393, -2.3466036, -4.7347169, -2.3376522, -1.9709382, 1.9840147
7: -15.7467308, -12.9583788, -15.7401857, -12.9499035, -2.6129894, 2.5918117
8: -0.7583771, 0.9028707, -0.7926233, 0.9181724, -1.3220387, 1.3415654
9: -6.6679935, -5.0688710, -6.6729708, -5.0150967, -1.6528969, 1.6040998

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9418325
time: 5.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9614651
time: 4.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.1566582, -10.5510864, -2.0893545, 2.0547032
1: -11.3276834, -8.4670172, -11.2935362, -8.4465971, -2.4580245, 2.4097364
2: -10.7271967, -8.5144558, -10.6638451, -8.5526924, -2.0876553, 2.1005268
3: -4.4033794, -2.3413148, -4.4218569, -2.3489740, -1.9147320, 1.9251730
4: -15.1088705, -12.5990896, -15.1383734, -12.5253992, -2.2541409, 2.2094111
5: 8.2397890, 9.6893950, 8.2301731, 9.6845121, -1.2460792, 1.2748141
6: -4.7403202, -2.3233070, -4.7347169, -2.3376522, -2.0203285, 2.0234203
7: -15.7568035, -12.9566870, -15.7401857, -12.9499035, -2.6170330, 2.5907478
8: -0.7772584, 0.9293244, -0.7926233, 0.9181724, -1.3326654, 1.3612602
9: -6.6664100, -5.0656857, -6.6729708, -5.0150967, -1.6513133, 1.6072850

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9478620
time: 3.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674946
time: 3.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.1864090, -10.5393200, -2.0685663, 2.0755379
1: -11.3179722, -8.4517279, -11.3032475, -8.4618835, -2.4307456, 2.4370501
2: -10.6882601, -8.5720196, -10.7027826, -8.4952068, -2.1229997, 2.0650618
3: -4.3971014, -2.3447456, -4.4281187, -2.3455315, -1.9118028, 1.9281054
4: -15.1120892, -12.5913963, -15.1351528, -12.5331144, -2.2503748, 2.2131805
5: 8.2523460, 9.6852093, 8.2176399, 9.6887217, -1.2493330, 1.2715828
6: -4.7101393, -2.3466036, -4.7648597, -2.3143473, -2.0103583, 2.0333683
7: -15.7467308, -12.9583788, -15.7503281, -12.9482136, -2.6119242, 2.5959463
8: -0.7583771, 0.9028707, -0.8115044, 0.9446242, -1.3417349, 1.3521813
9: -6.6679935, -5.0688710, -6.6713901, -5.0119200, -1.6560736, 1.6025190

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9416505
time: 3.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612830
time: 3.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.1864090, -10.5393200, -2.0665212, 2.0527053
1: -11.3276834, -8.4670172, -11.3032475, -8.4618835, -2.3864460, 2.3654714
2: -10.7271967, -8.5144558, -10.7027826, -8.4952068, -2.0681443, 2.0456645
3: -4.4033794, -2.3413148, -4.4281187, -2.3455315, -1.9161282, 1.9294910
4: -15.1088705, -12.5990896, -15.1351528, -12.5331144, -2.2608819, 2.2199295
5: 8.2397890, 9.6893950, 8.2176399, 9.6887217, -1.2589359, 1.2844211
6: -4.7403202, -2.3233070, -4.7648597, -2.3143473, -1.9930100, 2.0060375
7: -15.7568035, -12.9566870, -15.7503281, -12.9482136, -2.6160574, 2.5949717
8: -0.7772584, 0.9293244, -0.8115044, 0.9446242, -1.3580093, 1.3775268
9: -6.6664100, -5.0656857, -6.6713901, -5.0119200, -1.6544900, 1.6057043

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9484248
time: 3.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9680574
time: 3.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.1632938, -10.5476933, -2.0577598, 2.0369146
1: -11.2934866, -8.4585981, -11.3238506, -8.4201937, -2.4184480, 2.3754411
2: -10.6638260, -8.5663605, -10.7488365, -8.5667019, -2.0086827, 2.1087718
3: -4.4163847, -2.3490036, -4.3989482, -2.3426192, -1.9268274, 1.9091103
4: -15.1383724, -12.5146065, -15.1122799, -12.5839281, -2.2226748, 2.2685227
5: 8.2280121, 9.6844997, 8.2515297, 9.7067852, -1.2865638, 1.2451351
6: -4.7367697, -2.3376508, -4.7121048, -2.3158014, -2.0301771, 1.9726522
7: -15.7401762, -12.9545841, -15.7507696, -12.9455805, -2.6030579, 2.6077476
8: -0.7961533, 0.9181619, -0.7822597, 0.9031215, -1.3512521, 1.3466078
9: -6.6729279, -5.0165362, -6.6765742, -5.0665269, -1.6064010, 1.6600380

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9428800
time: 5.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9428825
time: 3.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.1631575, -10.5481625, -2.1138282, 2.0486002
1: -11.3031998, -8.4738846, -11.3237076, -8.4252548, -2.4914918, 2.3685980
2: -10.7027636, -8.5088758, -10.7469082, -8.5667763, -2.0307186, 2.2012663
3: -4.4226265, -2.3455589, -4.3989224, -2.3427069, -1.9332647, 1.9117980
4: -15.1351519, -12.5223141, -15.1122723, -12.5854073, -2.2176352, 2.2693079
5: 8.2154884, 9.6887102, 8.2515602, 9.7052383, -1.2918335, 1.2578962
6: -4.7669148, -2.3143487, -4.7120304, -2.3168576, -2.0796995, 1.9903564
7: -15.7503185, -12.9529018, -15.7507153, -12.9467287, -2.6059952, 2.6066060
8: -0.8150344, 0.9446130, -0.7812953, 0.9031134, -1.3843474, 1.3652133
9: -6.6713448, -5.0133581, -6.6755514, -5.0665808, -1.6047640, 1.6621933

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9490937
time: 4.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9496567
time: 3.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.1612816, -10.5356369, -2.0605102, 2.0266197
1: -11.2934866, -8.4585981, -11.2993670, -8.4270630, -2.4201765, 2.3686087
2: -10.6638260, -8.5663605, -10.7244101, -8.5612507, -1.9854088, 2.0559595
3: -4.4163847, -2.3490036, -4.4181337, -2.3469021, -1.9088435, 1.9076216
4: -15.1383724, -12.5146065, -15.1385632, -12.5071354, -2.2400231, 2.2364221
5: 8.2280121, 9.6844997, 8.2271652, 9.7061424, -1.2788388, 1.2629814
6: -4.7367697, -2.3376508, -4.7387552, -2.3068533, -2.0116296, 1.9722540
7: -15.7401762, -12.9545841, -15.7444134, -12.9417791, -2.6252937, 2.6175618
8: -0.7961533, 0.9181619, -0.8200352, 0.9184160, -1.3263354, 1.3475937
9: -6.6729279, -5.0165362, -6.6815081, -5.0141830, -1.6587448, 1.6649718

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613272, upper bound: 0.9421965
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613272, upper bound: 0.9421965
time: 3.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.1611528, -10.5361061, -2.1165304, 2.0383065
1: -11.3031998, -8.4738846, -11.2992249, -8.4321251, -2.4931850, 2.3617654
2: -10.7027636, -8.5088758, -10.7224770, -8.5613213, -2.0074499, 2.1485286
3: -4.4226265, -2.3455589, -4.4181099, -2.3469872, -1.9152956, 1.9102991
4: -15.1351519, -12.5223141, -15.1385574, -12.5086126, -2.2349706, 2.2372289
5: 8.2154884, 9.6887102, 8.2271986, 9.7045956, -1.2841327, 1.2757211
6: -4.7669148, -2.3143487, -4.7386794, -2.3079100, -2.0611873, 1.9899447
7: -15.7503185, -12.9529018, -15.7443552, -12.9429245, -2.6281433, 2.6164308
8: -0.8150344, 0.9446130, -0.8190703, 0.9184084, -1.3594413, 1.3661964
9: -6.6713448, -5.0133581, -6.6804857, -5.0142360, -1.6571088, 1.6671276

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613270, upper bound: 0.9484059
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613270, upper bound: 0.9489707
time: 3.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.3020706, -10.5062084, -2.0608387, 2.1236982
1: -11.3179722, -8.4517279, -11.4193640, -8.4549103, -2.4078851, 2.5414195
2: -10.6882601, -8.5720196, -10.7133293, -8.5639191, -2.0331559, 2.0890326
3: -4.3971014, -2.3447456, -4.5735259, -2.3432255, -1.9058156, 1.9523621
4: -15.1120892, -12.5913963, -15.1517277, -12.4463320, -2.1728892, 2.2093618
5: 8.2523460, 9.6852093, 8.2252398, 9.6461134, -1.2422373, 1.3221297
6: -4.7101393, -2.3466036, -4.7849193, -2.3436356, -1.9680562, 1.9907990
7: -15.7467308, -12.9583788, -15.7843103, -12.9958458, -2.3449669, 2.6853371
8: -0.7583771, 0.9028707, -0.7661648, 0.9712572, -1.4557819, 1.3398204
9: -6.6679935, -5.0688710, -6.6707277, -4.9604282, -1.7075653, 1.3997467

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9315604
time: 3.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9511930
time: 3.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.3020706, -10.5062084, -2.1167192, 2.1547587
1: -11.3276834, -8.4670172, -11.4193640, -8.4549103, -2.4813199, 2.5875554
2: -10.7271967, -8.5144558, -10.7133293, -8.5639191, -2.0912070, 2.1826668
3: -4.4033794, -2.3413148, -4.5735259, -2.3432255, -1.9123340, 1.9556813
4: -15.1088705, -12.5990896, -15.1517277, -12.4463320, -2.1776028, 2.2019448
5: 8.2397890, 9.6893950, 8.2252398, 9.6461134, -1.2525067, 1.3312576
6: -4.7403202, -2.3233070, -4.7849193, -2.3436356, -2.0174465, 2.0308790
7: -15.7568035, -12.9566870, -15.7843103, -12.9958458, -2.3549061, 2.6842732
8: -0.7772584, 0.9293244, -0.7661648, 0.9712572, -1.4664087, 1.3628532
9: -6.6664100, -5.0656857, -6.6707277, -4.9604282, -1.7059817, 1.4225795

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9375878
time: 3.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572203
time: 3.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.3320189, -10.4944439, -2.0959320, 2.1748428
1: -11.3179722, -8.4517279, -11.4292145, -8.4701881, -2.4540358, 2.6154852
2: -10.6882601, -8.5720196, -10.7522383, -8.5062780, -2.1269474, 2.1472068
3: -4.3971014, -2.3447456, -4.5798178, -2.3397806, -1.9092951, 1.9578025
4: -15.1120892, -12.5913963, -15.1485243, -12.4540434, -2.1738300, 2.2060046
5: 8.2523460, 9.6852093, 8.2126656, 9.6503067, -1.2548096, 1.3289509
6: -4.7101393, -2.3466036, -4.8156328, -2.3203645, -2.0074620, 2.0360017
7: -15.7467308, -12.9583788, -15.7946587, -12.9941196, -2.3494501, 2.6897950
8: -0.7583771, 0.9028707, -0.7849360, 0.9978013, -1.4756556, 1.3533471
9: -6.6679935, -5.0688710, -6.6690588, -4.9572344, -1.7107592, 1.4104955

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9313763
time: 3.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9510089
time: 3.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.3320189, -10.4944439, -2.0938869, 2.1522710
1: -11.3276834, -8.4670172, -11.4292145, -8.4701881, -2.4097395, 2.5437543
2: -10.7271967, -8.5144558, -10.7522383, -8.5062780, -2.0722909, 2.1278133
3: -4.4033794, -2.3413148, -4.5798178, -2.3397806, -1.9136195, 1.9595962
4: -15.1088705, -12.5990896, -15.1485243, -12.4540434, -2.1751475, 2.2124598
5: 8.2397890, 9.6893950, 8.2126656, 9.6503067, -1.2601626, 1.3407162
6: -4.7403202, -2.3233070, -4.8156328, -2.3203645, -1.9901109, 2.0170093
7: -15.7568035, -12.9566870, -15.7946587, -12.9941196, -2.3607659, 2.6888208
8: -0.7772584, 0.9293244, -0.7849360, 0.9978013, -1.4917140, 1.3738899
9: -6.6664100, -5.0656857, -6.6690588, -4.9572344, -1.7091756, 1.4135933

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9381503
time: 3.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9577831
time: 4.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.3090010, -10.5028172, -2.0851269, 2.1394362
1: -11.2934866, -8.4585981, -11.4499931, -8.4285011, -2.4417410, 2.5540519
2: -10.6638260, -8.5663605, -10.7983446, -8.5783138, -2.0131035, 2.1908941
3: -4.4163847, -2.3490036, -4.5506554, -2.3368518, -1.9247489, 1.9366243
4: -15.1383724, -12.5146065, -15.1256409, -12.5049629, -2.1416440, 2.2610283
5: 8.2280121, 9.6844997, 8.2465858, 9.6683960, -1.2920961, 1.3015654
6: -4.7367697, -2.3376508, -4.7620850, -2.3217676, -2.0272989, 1.9797480
7: -15.7401762, -12.9545841, -15.7955828, -12.9915333, -2.3387899, 2.7023888
8: -0.7961533, 0.9181619, -0.7559106, 0.9562240, -1.4849906, 1.3487785
9: -6.6729279, -5.0165362, -6.6744580, -5.0117989, -1.6611290, 1.4570351

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9326057
time: 5.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9326082
time: 3.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.3088646, -10.5032864, -2.1411948, 2.1506953
1: -11.3031998, -8.4738846, -11.4498463, -8.4335632, -2.5147848, 2.5471883
2: -10.7027636, -8.5088758, -10.7964144, -8.5783825, -2.0353737, 2.2833896
3: -4.4226265, -2.3455589, -4.5506287, -2.3369343, -1.9311695, 1.9397757
4: -15.1351519, -12.5223141, -15.1256332, -12.5064383, -2.1450319, 2.2618086
5: 8.2154884, 9.6887102, 8.2466173, 9.6668530, -1.3016806, 1.3143252
6: -4.7669148, -2.3143487, -4.7620201, -2.3228233, -2.0768189, 2.0016434
7: -15.7503185, -12.9529018, -15.7955198, -12.9926767, -2.3484936, 2.7012329
8: -0.8150344, 0.9446130, -0.7549405, 0.9562154, -1.5180864, 1.3710390
9: -6.6713448, -5.0133581, -6.6734362, -5.0118523, -1.6594925, 1.4796429

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9388197
time: 3.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9393824
time: 3.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.3068628, -10.4907560, -2.0878768, 2.1203675
1: -11.2934866, -8.4585981, -11.4255104, -8.4353676, -2.4434690, 2.5472212
2: -10.6638260, -8.5663605, -10.7739153, -8.5727987, -1.9900293, 2.1380818
3: -4.4163847, -2.3490036, -4.5699434, -2.3411362, -1.9067588, 1.9424818
4: -15.1383724, -12.5146065, -15.1519175, -12.4280653, -2.1771431, 2.2289264
5: 8.2280121, 9.6844997, 8.2221508, 9.6677227, -1.2786686, 1.3194152
6: -4.7367697, -2.3376508, -4.7887592, -2.3128185, -2.0087509, 1.9836667
7: -15.7401762, -12.9545841, -15.7891741, -12.9878426, -2.3682556, 2.7122025
8: -0.7961533, 0.9181619, -0.7936804, 0.9715168, -1.4600749, 1.3549984
9: -6.6729279, -5.0165362, -6.6792703, -4.9594526, -1.7134752, 1.4031022

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9319222
time: 3.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9319222
time: 3.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.3067322, -10.4912262, -2.1438966, 2.1316264
1: -11.3031998, -8.4738846, -11.4253626, -8.4404325, -2.5164776, 2.5403581
2: -10.7027636, -8.5088758, -10.7719860, -8.5728626, -2.0123048, 2.2306519
3: -4.4226265, -2.3455589, -4.5699196, -2.3412189, -1.9131961, 1.9456308
4: -15.1351519, -12.5223141, -15.1519127, -12.4295397, -2.1805477, 2.2297282
5: 8.2154884, 9.6887102, 8.2221851, 9.6661768, -1.2882247, 1.3321540
6: -4.7669148, -2.3143487, -4.7886901, -2.3138757, -2.0583067, 2.0057120
7: -15.7503185, -12.9529018, -15.7891102, -12.9889879, -2.3779249, 2.7110586
8: -0.8150344, 0.9446130, -0.7927108, 0.9715083, -1.4931803, 1.3772686
9: -6.6713448, -5.0133581, -6.6782479, -4.9595084, -1.7118363, 1.4257150

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9381337
time: 3.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9386964
time: 3.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.3041668, -10.5235815, -13.1566582, -10.5510864, -2.1321707, 2.0469761
1: -11.4438000, -8.4600401, -11.2935362, -8.4465971, -2.5624423, 2.3868756
2: -10.7377462, -8.5833111, -10.6638451, -8.5526924, -2.1116188, 2.0108461
3: -4.5487471, -2.3389950, -4.4218569, -2.3489740, -1.9352198, 1.9192486
4: -15.1254501, -12.5124207, -15.1383734, -12.5253992, -2.2503557, 2.1314657
5: 8.2474213, 9.6468496, 8.2301731, 9.6845121, -1.2966213, 1.2653879
6: -4.7603240, -2.3525896, -4.7347169, -2.3376522, -1.9783614, 1.9811292
7: -15.7909136, -13.0042801, -15.7401857, -12.9499035, -2.7065849, 2.3259325
8: -0.7319293, 0.9559593, -0.7926233, 0.9181724, -1.3222165, 1.4753087
9: -6.6658897, -5.0142002, -6.6729708, -5.0150967, -1.4295340, 1.6587706

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9175295
time: 5.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9371619
time: 5.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.3339596, -10.5118170, -13.1566582, -10.5510864, -2.1833563, 2.0820694
1: -11.4536533, -8.4753218, -11.2935362, -8.4465971, -2.6364765, 2.4330282
2: -10.7766523, -8.5255909, -10.6638451, -8.5526924, -2.1698015, 2.1047604
3: -4.5550556, -2.3355613, -4.4218569, -2.3489740, -1.9406986, 1.9227190
4: -15.1222477, -12.5201187, -15.1383734, -12.5253992, -2.2469873, 2.1324129
5: 8.2348223, 9.6510172, 8.2301731, 9.6845121, -1.3034432, 1.2779075
6: -4.7910752, -2.3293247, -4.7347169, -2.3376522, -2.0236661, 2.0205204
7: -15.8011827, -13.0025578, -15.7401857, -12.9499035, -2.7109461, 2.3303943
8: -0.7507010, 0.9825034, -0.7926233, 0.9181724, -1.3357446, 1.4951811
9: -6.6642218, -5.0110049, -6.6729708, -5.0150967, -1.4402833, 1.6619658

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9235564
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
time: 4.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.3041668, -10.5235815, -13.1864090, -10.5393200, -2.1632314, 2.1029043
1: -11.4438000, -8.4600401, -11.3032475, -8.4618835, -2.6085806, 2.4603443
2: -10.7377462, -8.5833111, -10.7027826, -8.4952068, -2.2051401, 2.0688899
3: -4.5487471, -2.3389950, -4.4281187, -2.3455315, -1.9385414, 1.9257593
4: -15.1254501, -12.5124207, -15.1351528, -12.5331144, -2.2429304, 2.1361618
5: 8.2474213, 9.6468496, 8.2176399, 9.6887217, -1.3057705, 1.2756877
6: -4.7603240, -2.3525896, -4.7648597, -2.3143473, -2.0182934, 2.0304830
7: -15.7909136, -13.0042801, -15.7503281, -12.9482136, -2.7055197, 2.3359084
8: -0.7319293, 0.9559593, -0.8115044, 0.9446242, -1.3452384, 1.4859247
9: -6.6658897, -5.0142002, -6.6713901, -5.0119200, -1.4523578, 1.6571898

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9173474
time: 3.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
time: 3.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.3339596, -10.5118170, -13.1864090, -10.5393200, -2.1607838, 2.0800719
1: -11.4536533, -8.4753218, -11.3032475, -8.4618835, -2.5647445, 2.3887658
2: -10.7766523, -8.5255909, -10.7027826, -8.4952068, -2.1502938, 2.0500968
3: -4.5550556, -2.3355613, -4.4281187, -2.3455315, -1.9424953, 1.9270368
4: -15.1222477, -12.5201187, -15.1351528, -12.5331144, -2.2534328, 2.1337142
5: 8.2348223, 9.6510172, 8.2176399, 9.6887217, -1.3152261, 1.2832954
6: -4.7910752, -2.3293247, -4.7648597, -2.3143473, -2.0045240, 2.0031343
7: -15.8011827, -13.0025578, -15.7503281, -12.9482136, -2.7099705, 2.3417449
8: -0.7507010, 0.9825034, -0.8115044, 0.9446242, -1.3562744, 1.5112302
9: -6.6642218, -5.0110049, -6.6713901, -5.0119200, -1.4433713, 1.6603851

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9241213
time: 3.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9437539
time: 3.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3020535, -10.5115213, -13.1632938, -10.5476933, -2.1578231, 2.0642803
1: -11.4193163, -8.4669085, -11.3238506, -8.4201937, -2.5962749, 2.3987362
2: -10.7133112, -8.5775843, -10.7488365, -8.5667019, -2.0908217, 2.1118913
3: -4.5681343, -2.3432529, -4.3989482, -2.3426192, -1.9518547, 1.9067128
4: -15.1517258, -12.4355240, -15.1122799, -12.5839281, -2.2152085, 2.1800570
5: 8.2230186, 9.6461086, 8.2515297, 9.7067852, -1.3430392, 1.2470007
6: -4.7869792, -2.3436360, -4.7121048, -2.3158014, -2.0359242, 1.9697707
7: -15.7843018, -13.0005989, -15.7507696, -12.9455805, -2.6965828, 2.3412304
8: -0.7696996, 0.9712491, -0.7822597, 0.9031215, -1.3481848, 1.4803512
9: -6.6707015, -4.9618669, -6.6765742, -5.0665269, -1.4014540, 1.7147074

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9185795
time: 3.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9185778
time: 5.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3320036, -10.4997549, -13.1631575, -10.5481625, -2.2090988, 2.0759671
1: -11.4291677, -8.4821873, -11.3237076, -8.4252548, -2.6699181, 2.3918920
2: -10.7522211, -8.5199451, -10.7469082, -8.5667763, -2.1128676, 2.2046924
3: -4.5744066, -2.3398066, -4.3989224, -2.3427069, -1.9572072, 1.9092917
4: -15.1485214, -12.4432383, -15.1122723, -12.5854073, -2.2104602, 2.1800895
5: 8.2104530, 9.6503019, 8.2515602, 9.7052383, -1.3492335, 1.2591749
6: -4.8176918, -2.3203642, -4.7120304, -2.3168576, -2.0813313, 1.9874580
7: -15.7946510, -12.9988842, -15.7507153, -12.9467287, -2.6998444, 2.3456321
8: -0.7884698, 0.9977911, -0.7812953, 0.9031134, -1.3797092, 1.4991345
9: -6.6690321, -4.9586730, -6.6755514, -5.0665808, -1.4044561, 1.7168784

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9247905
time: 3.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9253533
time: 3.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3020535, -10.5115213, -13.1612816, -10.5356369, -2.1501853, 2.0539861
1: -11.4193163, -8.4669085, -11.2993670, -8.4270630, -2.5980234, 2.3919036
2: -10.7133112, -8.5775843, -10.7244101, -8.5612507, -2.0675478, 2.0600336
3: -4.5681343, -2.3432529, -4.4181337, -2.3469021, -1.9429231, 1.9052694
4: -15.1517258, -12.4355240, -15.1385632, -12.5071354, -2.2325773, 2.1692519
5: 8.2230186, 9.6461086, 8.2271652, 9.7061424, -1.3352808, 1.2603652
6: -4.7869792, -2.3436360, -4.7387552, -2.3068533, -2.0228705, 1.9693689
7: -15.7843018, -13.0005989, -15.7444134, -12.9417791, -2.7188873, 2.3566282
8: -0.7696996, 0.9712491, -0.8200352, 0.9184160, -1.3313019, 1.4813366
9: -6.6707015, -4.9618669, -6.6815081, -5.0141830, -1.3904784, 1.7196412

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510529, upper bound: 0.9178911
time: 4.31 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510529, upper bound: 0.9178934
time: 3.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3320036, -10.4997549, -13.1611528, -10.5361061, -2.2014999, 2.0656734
1: -11.4291677, -8.4821873, -11.2992249, -8.4321251, -2.6716299, 2.3850594
2: -10.7522211, -8.5199451, -10.7224770, -8.5613213, -2.0895989, 2.1529188
3: -4.5744066, -2.3398066, -4.4181099, -2.3469872, -1.9483032, 1.9078369
4: -15.1485214, -12.4432383, -15.1385574, -12.5086126, -2.2278152, 2.1692996
5: 8.2104530, 9.6503019, 8.2271986, 9.7045956, -1.3414996, 1.2724864
6: -4.8176918, -2.3203642, -4.7386794, -2.3079100, -2.0683732, 1.9870420
7: -15.7946510, -12.9988842, -15.7443552, -12.9429245, -2.7220554, 2.3610196
8: -0.7884698, 0.9977911, -0.8190703, 0.9184084, -1.3628829, 1.5001171
9: -6.6690321, -4.9586730, -6.6804857, -5.0142360, -1.3934810, 1.7218127

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510526, upper bound: 0.9241046
time: 3.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510526, upper bound: 0.9246672
time: 3.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.64 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9418325
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9614651
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9478620
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674946
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9416505
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612830
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9484248
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9680574
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9428800
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9428825
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9490937
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9496567
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9613272, upper bound: 0.9421965
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9613272, upper bound: 0.9421965
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9613270, upper bound: 0.9484059
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9613270, upper bound: 0.9489707
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9315604
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9511930
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9375878
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572203
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9313763
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9510089
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9381503
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9577831
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9326057
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9326082
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9388197
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9369781, upper bound: 0.9393824
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9319222
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9319222
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9381337
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9386964
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9175295
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9371619
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9235564
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9173474
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9241213
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9437539
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9185795
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9185778
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9247905
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9253533
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510529, upper bound: 0.9178911
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510529, upper bound: 0.9178934
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510526, upper bound: 0.9241046
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.64
Output dim: 5, lower bound: -0.9510526, upper bound: 0.9246672
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9369801
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9369801
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9437539
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9173474
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9437519, upper bound: 0.9241212
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9178934
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.64
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9246672
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.3380417823791504
rel_dist={5: [-1.0100506396824613, 1.010049981815225]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7611974
time: 3.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
time: 3.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.60
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7611974
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.60
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.1630363, -10.4636698, -1.7673178, 1.7224104
1: -11.2995148, -8.4136467, -11.3005047, -8.3518839, -2.0647445, 2.0236039
2: -10.7255239, -8.5416927, -10.7255383, -8.5171347, -1.8689809, 1.8423712
3: -4.4290442, -2.3468359, -4.4308281, -2.2712052, -1.7046008, 1.6718259
4: -15.1385698, -12.5005589, -15.1814785, -12.4986553, -1.8990479, 1.9163349
5: 8.2217073, 9.7064924, 8.2192602, 9.7237635, -1.1317589, 1.1107540
6: -4.7417126, -2.3063393, -4.7438092, -2.2761712, -1.7519121, 1.7244308
7: -15.7444887, -12.9351664, -15.7447615, -12.9015751, -2.3000317, 2.2579718
8: -0.8229923, 0.9184313, -0.8540959, 0.9188125, -1.1811652, 1.2392077
9: -6.6816912, -5.0027695, -6.7087741, -5.0025487, -1.6048036, 1.6319242

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
time: 4.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
time: 3.87 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3069639, -10.4817972, -13.1611977, -10.4716988, -1.8661880, 1.7834792
1: -11.4256668, -8.4219580, -11.2992468, -8.3757257, -2.2326927, 2.0618978
2: -10.7750311, -8.5532455, -10.7255001, -8.5286980, -1.9428301, 1.8451490
3: -4.5807705, -2.3410683, -4.4293098, -2.2901108, -1.7237911, 1.6911969
4: -15.1519251, -12.4215012, -15.1726646, -12.4999018, -1.9043379, 1.8228164
5: 8.2167139, 9.6680660, 8.2213888, 9.7083225, -1.1852322, 1.0907634
6: -4.7917042, -2.3123016, -4.7414055, -2.2867792, -1.7334769, 1.7436256
7: -15.7892628, -12.9812613, -15.7443609, -12.9263458, -2.3794575, 1.9718959
8: -0.7966328, 0.9715328, -0.8333356, 0.9182534, -1.1516526, 1.3723288
9: -6.6794314, -4.9480395, -6.7000766, -5.0027800, -1.2938626, 1.6822457

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7369449
time: 4.03 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7458997
time: 3.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.64 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.64
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.64
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.64
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7369449
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.64
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7458997

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.1613884, -10.5266781, -1.7211914, 1.7211914
1: -11.2995148, -8.4136467, -11.2995148, -8.4136467, -2.0226507, 2.0226502
2: -10.7255239, -8.5416927, -10.7255239, -8.5416927, -1.8413625, 1.8413625
3: -4.4290442, -2.3468359, -4.4290442, -2.3468359, -1.6712050, 1.6712050
4: -15.1385698, -12.5005589, -15.1385698, -12.5005589, -1.8975039, 1.8975043
5: 8.2217073, 9.7064924, 8.2217073, 9.7064924, -1.1045458, 1.1045458
6: -4.7417126, -2.3063393, -4.7417126, -2.3063393, -1.7226939, 1.7226942
7: -15.7444887, -12.9351664, -15.7444887, -12.9351664, -2.2501545, 2.2501540
8: -0.8229923, 0.9184313, -0.8229923, 0.9184313, -1.1803603, 1.1803602
9: -6.6816912, -5.0027695, -6.6816912, -5.0027695, -1.5986805, 1.5986805

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7369430, upper bound: 0.7543090
time: 4.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7543108
time: 3.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.3069639, -10.4817972, -1.7485576, 1.8128595
1: -11.2995148, -8.4136467, -11.4256668, -8.4219580, -2.0459442, 2.2012644
2: -10.7255239, -8.5416927, -10.7750311, -8.5532455, -1.8449540, 1.9234838
3: -4.4290442, -2.3468359, -4.5807705, -2.3410683, -1.6690779, 1.6819389
4: -15.1385698, -12.5005589, -15.1519251, -12.4215012, -1.8005939, 1.8899868
5: 8.2217073, 9.7064924, 8.2167139, 9.6680660, -1.1108122, 1.1609823
6: -4.7417126, -2.3063393, -4.7917042, -2.3123016, -1.7198186, 1.7054846
7: -15.7444887, -12.9351664, -15.7892628, -12.9812613, -1.9669504, 2.3447490
8: -0.8229923, 0.9184313, -0.7966328, 0.9715328, -1.3140988, 1.1799984
9: -6.6816912, -5.0027695, -6.6794314, -4.9480395, -1.6564837, 1.2636719

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7369430, upper bound: 0.7543108
time: 3.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7543109
time: 3.95 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.3069515, -10.4884186, -13.1631622, -10.4921703, -1.8421488, 1.7734592
1: -11.4256506, -8.4239092, -11.3236790, -8.3817720, -2.2277141, 2.0770271
2: -10.7750282, -8.5650015, -10.7499084, -8.5536308, -1.8914862, 1.8066382
3: -4.5693364, -2.3410773, -4.3992453, -2.2858534, -1.6996636, 1.6548595
4: -15.1519241, -12.4581127, -15.1463795, -12.5831299, -1.8287597, 1.7765183
5: 8.2320738, 9.6680660, 8.2512188, 9.7089529, -1.1659660, 1.0522621
6: -4.7810769, -2.3123050, -4.7118235, -2.2957265, -1.7113690, 1.7110033
7: -15.7892590, -12.9846077, -15.7507114, -12.9365740, -2.3661203, 1.9822295
8: -0.7835765, 0.9715288, -0.7929790, 0.9029474, -1.1214180, 1.3283763
9: -6.6794281, -4.9692764, -6.6950769, -5.0665016, -1.2237215, 1.6745458

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226007
time: 3.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280818
time: 4.63 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3069630, -10.4829235, -13.1611624, -10.4801159, -1.8336072, 1.7824931
1: -11.4256582, -8.4233322, -11.2991924, -8.3886404, -2.2351069, 2.0604353
2: -10.7750292, -8.5553246, -10.7254810, -8.5481777, -1.8701029, 1.8446524
3: -4.5792122, -2.3410726, -4.4184198, -2.2901418, -1.7226286, 1.6481645
4: -15.1519251, -12.4221897, -15.1726627, -12.5063591, -1.8319674, 1.8210783
5: 8.2173834, 9.6680651, 8.2268314, 9.7083130, -1.1838560, 1.0576612
6: -4.7913933, -2.3123019, -4.7384815, -2.2867799, -1.7333708, 1.7055912
7: -15.7892599, -12.9819431, -15.7443523, -12.9327574, -2.3835487, 1.9711263
8: -0.7963567, 0.9715302, -0.8307548, 0.9182429, -1.1514480, 1.3223710
9: -6.6794286, -4.9492517, -6.7000256, -5.0141563, -1.2067552, 1.6807876

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7339849, upper bound: 0.7458980
time: 3.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7339849, upper bound: 0.7458999
time: 4.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7369430, upper bound: 0.7543090
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7543108
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7369430, upper bound: 0.7543108
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7543109
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226007
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280818
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7339849, upper bound: 0.7458980
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 5, lower bound: -0.7339849, upper bound: 0.7458999

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1633673, -10.5471478, -13.1613722, -10.5332966, -1.7111945, 1.7010469
1: -11.3239450, -8.4196949, -11.2994995, -8.4156008, -2.0377798, 2.0176752
2: -10.7499352, -8.5666199, -10.7255211, -8.5533848, -1.8022733, 1.7900221
3: -4.3989763, -2.3425825, -4.4175997, -2.3468444, -1.6348753, 1.6399078
4: -15.1122828, -12.5838108, -15.1385651, -12.5370502, -1.8468609, 1.8219337
5: 8.2515154, 9.7071228, 8.2370262, 9.7064877, -1.0675564, 1.0852680
6: -4.7121363, -2.3152876, -4.7310743, -2.3063416, -1.6900797, 1.6966984
7: -15.7508335, -12.9453793, -15.7444887, -12.9386206, -2.2554560, 2.2368207
8: -0.7826340, 0.9031253, -0.8099329, 0.9184294, -1.1364138, 1.1482214
9: -6.6767054, -5.0664907, -6.6816764, -5.0240040, -1.5909991, 1.5552197

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
time: 4.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7474756, upper bound: 0.7564734
time: 5.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1613827, -10.5278044, -1.7202063, 1.7018762
1: -11.2994614, -8.4265642, -11.2995081, -8.4150257, -2.0211883, 2.0250490
2: -10.7255049, -8.5611706, -10.7255230, -8.5437679, -1.8408613, 1.7686405
3: -4.4181614, -2.3468671, -4.4274831, -2.3468378, -1.6281366, 1.6701679
4: -15.1385651, -12.5070190, -15.1385689, -12.5012493, -1.8957014, 1.8251171
5: 8.2271509, 9.7064800, 8.2223759, 9.7064915, -1.0770149, 1.1031696
6: -4.7387886, -2.3063402, -4.7414012, -2.3063397, -1.6846704, 1.7225728
7: -15.7444801, -12.9415779, -15.7444878, -12.9358492, -2.2487907, 2.2541823
8: -0.8204088, 0.9184208, -0.8227153, 0.9184303, -1.1304138, 1.1802803
9: -6.6816416, -5.0141444, -6.6816854, -5.0039825, -1.5972223, 1.5834708

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7653423, upper bound: 0.7534467
time: 3.81 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7653423, upper bound: 0.7653444
time: 4.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1633673, -10.5471478, -13.3069515, -10.4884186, -1.7385607, 1.7888238
1: -11.3239450, -8.4196949, -11.4256506, -8.4239092, -2.0610733, 2.1962857
2: -10.7499352, -8.5666199, -10.7750282, -8.5650015, -1.8064446, 1.8721447
3: -4.3989763, -2.3425825, -4.5693364, -2.3410773, -1.6327491, 1.6577952
4: -15.1122828, -12.5838108, -15.1519241, -12.4581127, -1.7542882, 1.8144155
5: 8.2515154, 9.7071228, 8.2320738, 9.6680660, -1.0723565, 1.1417167
6: -4.7121363, -2.3152876, -4.7810769, -2.3123050, -1.6872034, 1.6833577
7: -15.7508335, -12.9453793, -15.7892590, -12.9846077, -1.9772820, 2.3314133
8: -0.7826340, 0.9031253, -0.7835765, 0.9715288, -1.2701526, 1.1497622
9: -6.6767054, -5.0664907, -6.6794281, -4.9692764, -1.6487999, 1.1935294

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7438394
time: 3.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7280801, upper bound: 0.7454480
time: 3.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.3069630, -10.4829235, -1.7475710, 1.7802832
1: -11.2994614, -8.4265642, -11.4256582, -8.4233322, -2.0444818, 2.2036793
2: -10.7255049, -8.5611706, -10.7750292, -8.5553246, -1.8444581, 1.8507624
3: -4.4181614, -2.3468671, -4.5792122, -2.3410726, -1.6260548, 1.6807742
4: -15.1385651, -12.5070190, -15.1519251, -12.4221897, -1.7988558, 1.8176208
5: 8.2271509, 9.7064800, 8.2173834, 9.6680651, -1.0777576, 1.1596063
6: -4.7387886, -2.3063402, -4.7913933, -2.3123019, -1.6817923, 1.7053790
7: -15.7444801, -12.9415779, -15.7892599, -12.9819431, -1.9661837, 2.3488417
8: -0.8204088, 0.9184208, -0.7963567, 0.9715302, -1.2641528, 1.1797948
9: -6.6816416, -5.0141444, -6.6794286, -4.9492517, -1.6550255, 1.1765630

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7423981
time: 4.02 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7543108
time: 3.72 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3020658, -10.5097313, -13.1618557, -10.4990234, -1.8272185, 1.7334847
1: -11.4193554, -8.4559507, -11.3220625, -8.3902826, -2.2063308, 2.0201614
2: -10.7133265, -8.5700626, -10.7312155, -8.5549717, -1.8174500, 1.7849216
3: -4.5674419, -2.3432286, -4.3987532, -2.2864716, -1.6977034, 1.6522911
4: -15.1517267, -12.4656668, -15.1463242, -12.5851212, -1.8243217, 1.7655380
5: 8.2329454, 9.6461153, 8.2514591, 9.7032127, -1.1543684, 1.0271252
6: -4.7792645, -2.3436372, -4.7112813, -2.3044972, -1.6988318, 1.6687522
7: -15.7843122, -12.9975567, -15.7496357, -12.9399853, -2.3580303, 1.9687595
8: -0.7592177, 0.9712560, -0.7865834, 0.9028730, -1.0939023, 1.3170023
9: -6.6707263, -4.9717278, -6.6928005, -5.0671344, -1.2086444, 1.6689587

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226006
time: 4.30 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226007
time: 3.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3320141, -10.4979601, -13.1625147, -10.4952259, -1.8873320, 1.7448583
1: -11.4292088, -8.4712267, -11.3229465, -8.3990116, -2.2933111, 2.0166044
2: -10.7522354, -8.5123501, -10.7405272, -8.5540829, -1.8359709, 1.8920369
3: -4.5737362, -2.3397839, -4.3990798, -2.2862315, -1.7032824, 1.6551690
4: -15.1485233, -12.4733810, -15.1463499, -12.5880680, -1.8183122, 1.7678742
5: 8.2203693, 9.6503067, 8.2513580, 9.7031145, -1.1644931, 1.0385848
6: -4.8099790, -2.3203669, -4.7114925, -2.3005981, -1.7539473, 1.6807263
7: -15.7946568, -12.9958296, -15.7503471, -12.9407406, -2.3609724, 1.9731839
8: -0.7779899, 0.9977987, -0.7901506, 0.9029086, -1.1188531, 1.3400011
9: -6.6690578, -4.9685330, -6.6914911, -5.0667801, -1.2123013, 1.6804528

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280820
time: 4.05 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3090773, -10.5022688, -13.1611624, -10.4801159, -1.8535004, 1.7633197
1: -11.4500942, -8.4280024, -11.2991924, -8.3886404, -2.2362294, 2.0568869
2: -10.7994423, -8.5782375, -10.7254810, -8.5481777, -1.9207540, 1.7944272
3: -4.5506849, -2.3368161, -4.4184198, -2.2901418, -1.6901503, 1.6712570
4: -15.1256447, -12.5048466, -15.1726627, -12.5063591, -1.8781700, 1.7509019
5: 8.2465706, 9.6687326, 8.2268314, 9.7083130, -1.1482320, 1.0789598
6: -4.7621164, -2.3212516, -4.7384815, -2.2867799, -1.7050614, 1.7291415
7: -15.7956581, -12.9913330, -15.7443523, -12.9327574, -2.3782654, 1.9645216
8: -0.7562866, 0.9562299, -0.8307548, 0.9182429, -1.1110106, 1.3542643
9: -6.6745906, -5.0117588, -6.7000256, -5.0141563, -1.2667241, 1.6387544

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7196407, upper bound: 0.7354266
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251220, upper bound: 0.7370367
time: 3.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.1611624, -10.4801159, -1.8335862, 1.7641189
1: -11.4256086, -8.4348717, -11.2991924, -8.3886404, -2.2350645, 2.0642557
2: -10.7750120, -8.5727234, -10.7254810, -8.5481777, -1.8700376, 1.7731895
3: -4.5699735, -2.3410995, -4.4184198, -2.2901418, -1.6901565, 1.6481047
4: -15.1519194, -12.4279480, -15.1726627, -12.5063591, -1.8319445, 1.7669394
5: 8.2221375, 9.6680603, 8.2268314, 9.7083130, -1.1576930, 1.0576571
6: -4.7887893, -2.3123035, -4.7384815, -2.2867799, -1.7035379, 1.7055674
7: -15.7892513, -12.9876451, -15.7443523, -12.9327574, -2.3835030, 1.9880104
8: -0.7940555, 0.9715207, -0.8307548, 0.9182429, -1.1086640, 1.3223424
9: -6.6794038, -4.9594150, -6.7000256, -5.0141563, -1.2067218, 1.6669855

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7196407, upper bound: 0.7235461
time: 3.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251220, upper bound: 0.7251549
time: 4.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7474756, upper bound: 0.7564734
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7653423, upper bound: 0.7534467
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7653423, upper bound: 0.7653444
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7438394
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7280801, upper bound: 0.7454480
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7423981
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7543108
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226006
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226007
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280820
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7196407, upper bound: 0.7354266
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7251220, upper bound: 0.7370367
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7196407, upper bound: 0.7235461
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 5, lower bound: -0.7251220, upper bound: 0.7251549

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1620617, -10.5540009, -13.1566496, -10.5546093, -1.6712213, 1.6863241
1: -11.3223324, -8.4282084, -11.2935276, -8.4476376, -1.9809122, 1.9970648
2: -10.7312403, -8.5680332, -10.6638432, -8.5587721, -1.7805865, 1.7159407
3: -4.3984842, -2.3431964, -4.4157672, -2.3489795, -1.6323414, 1.6382697
4: -15.1122246, -12.5858021, -15.1383734, -12.5446138, -1.8401570, 1.8174624
5: 8.2517548, 9.7013798, 8.2378788, 9.6845093, -1.0448306, 1.0736167
6: -4.7115936, -2.3240569, -4.7290587, -2.3376524, -1.6478248, 1.6839504
7: -15.7497559, -12.9487991, -15.7401867, -12.9516220, -2.2429380, 2.2297931
8: -0.7762470, 0.9030502, -0.7856748, 0.9181707, -1.1251485, 1.1222520
9: -6.6744304, -5.0671225, -6.6729627, -5.0263948, -1.5856218, 1.5443754

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7511269
time: 5.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1627216, -10.5502033, -13.1864004, -10.5428410, -1.6825938, 1.7512684
1: -11.3232145, -8.4369335, -11.3032408, -8.4629269, -1.9773555, 2.0834422
2: -10.7405539, -8.5670967, -10.7027807, -8.5012150, -1.8875661, 1.7344713
3: -4.3988109, -2.3429549, -4.4220285, -2.3455362, -1.6352382, 1.6450872
4: -15.1122541, -12.5887508, -15.1351528, -12.5523262, -1.8430119, 1.8111811
5: 8.2516556, 9.7012825, 8.2253466, 9.6887169, -1.0566434, 1.0828464
6: -4.7118077, -2.3201573, -4.7592015, -2.3143482, -1.6598077, 1.7428930
7: -15.7504702, -12.9495525, -15.7503281, -12.9499321, -2.2424545, 2.2324071
8: -0.7797508, 0.9030869, -0.8045552, 0.9446216, -1.1479456, 1.1482216
9: -6.6731215, -5.0667696, -6.6713810, -5.0232177, -1.5970044, 1.5449629

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7499435
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7459543
time: 4.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1633673, -10.5471478, -1.7010317, 1.7141893
1: -11.2994614, -8.4265642, -11.3239450, -8.4196949, -2.0176411, 2.0261760
2: -10.7255049, -8.5611706, -10.7499352, -8.5666199, -1.7899604, 1.8192854
3: -4.4181614, -2.3468671, -4.3989763, -2.3425825, -1.6512256, 1.6347556
4: -15.1385651, -12.5070190, -15.1122828, -12.5838108, -1.8218732, 1.8713117
5: 8.2271509, 9.7064800, 8.2515154, 9.7071228, -1.0930916, 1.0675511
6: -4.7387886, -2.3063402, -4.7121363, -2.3152876, -1.7082138, 1.6900339
7: -15.7444801, -12.9415779, -15.7508335, -12.9453793, -2.2367077, 2.2488875
8: -0.8204088, 0.9184208, -0.7826340, 0.9031253, -1.1622908, 1.1363862
9: -6.6816416, -5.0141444, -6.6767054, -5.0664907, -1.5551910, 1.5965304

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7549721, upper bound: 0.7392206
time: 6.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7564730, upper bound: 0.7445762
time: 4.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1613560, -10.5350914, -1.7018547, 1.7018549
1: -11.2994614, -8.4265642, -11.2994614, -8.4265642, -2.0250082, 2.0250084
2: -10.7255049, -8.5611706, -10.7255049, -8.5611706, -1.7685745, 1.7685747
3: -4.4181614, -2.3468671, -4.4181614, -2.3468671, -1.6280766, 1.6280768
4: -15.1385651, -12.5070190, -15.1385651, -12.5070190, -1.8250947, 1.8250947
5: 8.2271509, 9.7064800, 8.2271509, 9.7064800, -1.0770078, 1.0770079
6: -4.7387886, -2.3063402, -4.7387886, -2.3063402, -1.6846461, 1.6846461
7: -15.7444801, -12.9415779, -15.7444801, -12.9415779, -2.2541389, 2.2541389
8: -0.8204088, 0.9184208, -0.8204088, 0.9184208, -1.1303854, 1.1303852
9: -6.6816416, -5.0141444, -6.6816416, -5.0141444, -1.5834398, 1.5834398

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7549723, upper bound: 0.7392570
time: 5.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7564732, upper bound: 0.7446123
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1620617, -10.5540009, -13.3020658, -10.5097313, -1.6985860, 1.7738934
1: -11.3223324, -8.4282084, -11.4193554, -8.4559507, -2.0042067, 2.1749022
2: -10.7312403, -8.5680332, -10.7133265, -8.5700626, -1.7842631, 1.7980804
3: -4.3984842, -2.3431964, -4.5674419, -2.3432286, -1.6299434, 1.6558907
4: -15.1122246, -12.5858021, -15.1517267, -12.4656668, -1.7432961, 1.8099952
5: 8.2517548, 9.7013798, 8.2329454, 9.6461153, -1.0473347, 1.1300720
6: -4.7115936, -2.3240569, -4.7792645, -2.3436372, -1.6449428, 1.6708231
7: -15.7497559, -12.9487991, -15.7843122, -12.9975567, -1.9639034, 2.3233194
8: -0.7762470, 0.9030502, -0.7592177, 0.9712560, -1.2588916, 1.1217549
9: -6.6744304, -5.0671225, -6.6707263, -4.9717278, -1.6432133, 1.1784534

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7399667
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7438395
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1627216, -10.5502033, -13.3320141, -10.4979601, -1.7099595, 1.8340075
1: -11.3232145, -8.4369335, -11.4292088, -8.4712267, -2.0006504, 2.2618830
2: -10.7405539, -8.5670967, -10.7522354, -8.5123501, -1.8915520, 1.8166199
3: -4.3988109, -2.3429549, -4.5737362, -2.3397839, -1.6327300, 1.6614625
4: -15.1122541, -12.5887508, -15.1485233, -12.4733810, -1.7456379, 1.8040061
5: 8.2516556, 9.7012825, 8.2203693, 9.6503067, -1.0587566, 1.1402266
6: -4.7118077, -2.3201573, -4.8099790, -2.3203669, -1.6569090, 1.7259374
7: -15.7504702, -12.9495525, -15.7946568, -12.9958296, -1.9684443, 2.3262572
8: -0.7797508, 0.9030869, -0.7779899, 0.9977987, -1.2818661, 1.1464503
9: -6.6731215, -5.0667696, -6.6690578, -4.9685330, -1.6547074, 1.1821103

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7389715
time: 4.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7350132
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.3090773, -10.5022688, -1.7283993, 1.8001704
1: -11.2994614, -8.4265642, -11.4500942, -8.4280024, -2.0409336, 2.2048011
2: -10.7255049, -8.5611706, -10.7994423, -8.5782375, -1.7942333, 1.9014072
3: -4.4181614, -2.3468671, -4.5506849, -2.3368161, -1.6491508, 1.6482961
4: -15.1385651, -12.5070190, -15.1256447, -12.5048466, -1.7286797, 1.8638158
5: 8.2271509, 9.7064800, 8.2465706, 9.6687326, -1.0989974, 1.1239823
6: -4.7387886, -2.3063402, -4.7621164, -2.3212516, -1.7053342, 1.6770704
7: -15.7444801, -12.9415779, -15.7956581, -12.9913330, -1.9595790, 2.3435459
8: -0.8204088, 0.9184208, -0.7562866, 0.9562299, -1.2960300, 1.1393688
9: -6.6816416, -5.0141444, -6.6745906, -5.0117588, -1.6129923, 1.2365348

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354262, upper bound: 0.7280522
time: 4.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370349, upper bound: 0.7335351
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.3069401, -10.4902096, -1.7292204, 1.7802622
1: -11.2994614, -8.4265642, -11.4256086, -8.4348717, -2.0483017, 2.2036366
2: -10.7255049, -8.5611706, -10.7750120, -8.5727234, -1.7729974, 1.8506968
3: -4.4181614, -2.3468671, -4.5699735, -2.3410995, -1.6259966, 1.6482940
4: -15.1385651, -12.5070190, -15.1519194, -12.4279480, -1.7447104, 1.8175981
5: 8.2271509, 9.7064800, 8.2221375, 9.6680603, -1.0777535, 1.1334435
6: -4.7387886, -2.3063402, -4.7887893, -2.3123035, -1.6817675, 1.6755264
7: -15.7444801, -12.9415779, -15.7892513, -12.9876451, -1.9830627, 2.3487968
8: -0.8204088, 0.9184208, -0.7940555, 0.9715207, -1.2641242, 1.1370125
9: -6.6816416, -5.0141444, -6.6794038, -4.9594150, -1.6412406, 1.1765306

Time for backsubstitution: 5.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7280845
time: 5.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7335657
time: 4.22 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.3020658, -10.5097313, -13.1620617, -10.5540009, -1.7738934, 1.6985857
1: -11.4193554, -8.4559507, -11.3223324, -8.4282084, -2.1749020, 2.0042069
2: -10.7133265, -8.5700626, -10.7312403, -8.5680332, -1.7980804, 1.7842631
3: -4.5674419, -2.3432286, -4.3984842, -2.3431964, -1.6558907, 1.6299434
4: -15.1517267, -12.4656668, -15.1122246, -12.5858021, -1.8099952, 1.7432961
5: 8.2329454, 9.6461153, 8.2517548, 9.7013798, -1.1300721, 1.0473347
6: -4.7792645, -2.3436372, -4.7115936, -2.3240569, -1.6708231, 1.6449428
7: -15.7843122, -12.9975567, -15.7497559, -12.9487991, -2.3233194, 1.9639039
8: -0.7592177, 0.9712560, -0.7762470, 0.9030502, -1.1217546, 1.2588918
9: -6.6707263, -4.9717278, -6.6744304, -5.0671225, -1.1784534, 1.6432128

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226005
time: 4.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226005
time: 4.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.3020658, -10.5097313, -13.3077297, -10.5091209, -1.6880851, 1.6705527
1: -11.4193554, -8.4559507, -11.4483929, -8.4365149, -2.0338979, 2.0177786
2: -10.7133265, -8.5700626, -10.7807407, -8.5795660, -1.7151132, 1.7795634
3: -4.5674419, -2.3432286, -4.5501785, -2.3374286, -1.4611843, 1.4510810
4: -15.1517267, -12.4656668, -15.1255865, -12.5068340, -1.6496670, 1.6698134
5: 8.2329454, 9.6461153, 8.2468157, 9.6629982, -1.0438412, 1.0163990
6: -4.7792645, -2.3436372, -4.7616272, -2.3300266, -1.5882750, 1.5529072
7: -15.7843122, -12.9975567, -15.7944136, -12.9947395, -1.9614353, 1.9705818
8: -0.7592177, 0.9712560, -0.7498717, 0.9561501, -1.0904408, 1.0962042
9: -6.6707263, -4.9717278, -6.6723170, -5.0124073, -1.2084975, 1.2331917

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226006
time: 4.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226006
time: 4.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3320141, -10.4979601, -13.1627216, -10.5502033, -1.8340073, 1.7099597
1: -11.4292088, -8.4712267, -11.3232145, -8.4369335, -2.2618833, 2.0006499
2: -10.7522354, -8.5123501, -10.7405539, -8.5670967, -1.8166199, 1.8915522
3: -4.5737362, -2.3397839, -4.3988109, -2.3429549, -1.6614623, 1.6327300
4: -15.1485233, -12.4733810, -15.1122541, -12.5887508, -1.8040061, 1.7456379
5: 8.2203693, 9.6503067, 8.2516556, 9.7012825, -1.1402264, 1.0587567
6: -4.8099790, -2.3203669, -4.7118077, -2.3201573, -1.7259374, 1.6569088
7: -15.7946568, -12.9958296, -15.7504702, -12.9495525, -2.3262568, 1.9684441
8: -0.7779899, 0.9977987, -0.7797508, 0.9030869, -1.1464505, 1.2818661
9: -6.6690578, -4.9685330, -6.6731215, -5.0667696, -1.1821103, 1.6547079

Time for backsubstitution: 5.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280819
time: 4.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280807
time: 5.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3320141, -10.4979601, -13.3084154, -10.5053253, -1.7483406, 1.6801426
1: -11.4292088, -8.4712267, -11.4493256, -8.4452410, -2.1206512, 2.0142066
2: -10.7522354, -8.5123501, -10.7900562, -8.5786839, -1.7337737, 1.8866799
3: -4.5737362, -2.3397839, -4.5505123, -2.3371875, -1.4666510, 1.4544299
4: -15.1485233, -12.4733810, -15.1256161, -12.5097771, -1.6537216, 1.6720760
5: 8.2203693, 9.6503067, 8.2467136, 9.6629019, -1.0593251, 1.0278177
6: -4.8099790, -2.3203669, -4.7618217, -2.3261244, -1.6433144, 1.5674720
7: -15.7946568, -12.9958296, -15.7952347, -12.9955006, -1.9715424, 1.9750094
8: -0.7779899, 0.9977987, -0.7533553, 0.9561877, -1.1153345, 1.1242406
9: -6.6690578, -4.9685330, -6.6710076, -5.0120454, -1.2121487, 1.2584729

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280819
time: 3.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280820
time: 4.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3077297, -10.5091209, -13.1564312, -10.5014238, -1.8175578, 1.7485948
1: -11.4483929, -8.4365149, -11.2932129, -8.4206734, -2.1791763, 2.0362759
2: -10.7807407, -8.5795660, -10.6638031, -8.5530853, -1.8991919, 1.7204149
3: -4.5501785, -2.3374286, -4.4166384, -2.2922921, -1.6876459, 1.6696324
4: -15.1255865, -12.5068340, -15.1724663, -12.5139484, -1.8714743, 1.7456093
5: 8.2468157, 9.6629982, 8.2276936, 9.6863403, -1.1256697, 1.0665406
6: -4.7616272, -2.3300266, -4.7364626, -2.3180964, -1.6633229, 1.7163930
7: -15.7944136, -12.9947395, -15.7400417, -12.9457359, -2.3654976, 1.9596367
8: -0.7498717, 0.9561501, -0.8064628, 0.9179845, -1.0996556, 1.3278451
9: -6.6723170, -5.0124073, -6.6913118, -5.0165491, -1.2610409, 1.6278448

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7315554
time: 4.30 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7354263
time: 6.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3084154, -10.5053253, -13.1861954, -10.4896593, -1.8271866, 1.8135533
1: -11.4493256, -8.4452410, -11.3029346, -8.4359608, -2.1757483, 2.1226914
2: -10.7900562, -8.5786839, -10.7027388, -8.4956722, -2.0061383, 1.7390816
3: -4.5505123, -2.3371875, -4.4228849, -2.2888448, -1.6909642, 1.6764436
4: -15.1256161, -12.5097771, -15.1692419, -12.5216560, -1.8742986, 1.7494221
5: 8.2467136, 9.6629019, 8.2151632, 9.6905451, -1.1374282, 1.0811725
6: -4.7618217, -2.3261244, -4.7666445, -2.2947884, -1.6778612, 1.7753465
7: -15.7952347, -12.9955006, -15.7501850, -12.9440174, -2.3652554, 1.9697216
8: -0.7533553, 0.9561877, -0.8253829, 0.9444470, -1.1275744, 1.3535078
9: -6.6710076, -5.0120454, -6.6897416, -5.0133729, -1.2863109, 1.6284776

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7305603
time: 3.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7266021
time: 3.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3056011, -10.4970589, -13.1564312, -10.5014238, -1.7976446, 1.7492802
1: -11.4239092, -8.4433861, -11.2932129, -8.4206734, -2.1780105, 2.0436692
2: -10.7563124, -8.5739956, -10.6638031, -8.5530853, -1.8482690, 1.6991858
3: -4.5694919, -2.3417053, -4.4166384, -2.2922921, -1.6876383, 1.6464367
4: -15.1518612, -12.4299374, -15.1724663, -12.5139484, -1.8252296, 1.7616587
5: 8.2223921, 9.6623058, 8.2276936, 9.6863403, -1.1351881, 1.0452292
6: -4.7882972, -2.3210754, -4.7364626, -2.3180964, -1.6619170, 1.6928360
7: -15.7879543, -12.9910526, -15.7400417, -12.9457359, -2.3707428, 1.9832225
8: -0.7876420, 0.9714403, -0.8064628, 0.9179845, -1.0973167, 1.2959280
9: -6.6771307, -4.9600654, -6.6913118, -5.0165491, -1.2011275, 1.6560783

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7196712
time: 5.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7235437
time: 4.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3062897, -10.4932652, -13.1861954, -10.4896593, -1.8072729, 1.8141937
1: -11.4248438, -8.4521084, -11.3029346, -8.4359608, -2.1745830, 2.1300507
2: -10.7656260, -8.5731544, -10.7027388, -8.4956722, -1.9552903, 1.7178571
3: -4.5698099, -2.3414660, -4.4228849, -2.2888448, -1.6909552, 1.6532643
4: -15.1518917, -12.4328842, -15.1692419, -12.5216560, -1.8280764, 1.7654879
5: 8.2222862, 9.6622219, 8.2151632, 9.6905451, -1.1469259, 1.0598328
6: -4.7884893, -2.3171744, -4.7666445, -2.2947884, -1.6766052, 1.7518256
7: -15.7888126, -12.9918098, -15.7501850, -12.9440174, -2.3705101, 1.9932714
8: -0.7911248, 0.9714806, -0.8253829, 0.9444470, -1.1252463, 1.3216014
9: -6.6758208, -4.9597025, -6.6897416, -5.0133729, -1.2264018, 1.6567087

Time for backsubstitution: 5.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7269176, upper bound: 0.7172499
time: 4.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7269176, upper bound: 0.7175494
time: 3.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.79 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7511269
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7499435
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7459543
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7549721, upper bound: 0.7392206
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7564730, upper bound: 0.7445762
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7549723, upper bound: 0.7392570
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7564732, upper bound: 0.7446123
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7399667
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7438395
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7389715
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7350132
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7354262, upper bound: 0.7280522
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7370349, upper bound: 0.7335351
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7280845
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7335657
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226005
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226005
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226006
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7226006
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280819
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280807
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280819
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280820
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7315554
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7354263
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7305603
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7266021
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7196712
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7235437
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7269176, upper bound: 0.7172499
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 5, lower bound: -0.7269176, upper bound: 0.7175494

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.1566496, -10.5546093, -1.6680617, 1.6580255
1: -11.3179722, -8.4517279, -11.2935276, -8.4476376, -1.9784589, 1.9583290
2: -10.6882601, -8.5720196, -10.6638432, -8.5587721, -1.7270014, 1.7145469
3: -4.3971014, -2.3447456, -4.4157672, -2.3489795, -1.6315756, 1.6366096
4: -15.1120892, -12.5913963, -15.1383734, -12.5446138, -1.8384399, 1.8135176
5: 8.2523460, 9.6852093, 8.2378788, 9.6845093, -1.0411417, 1.0589129
6: -4.7101393, -2.3466036, -4.7290587, -2.3376524, -1.6465206, 1.6531084
7: -15.7467308, -12.9583788, -15.7401867, -12.9516220, -2.2402163, 2.2213311
8: -0.7583771, 0.9028707, -0.7856748, 0.9181707, -1.1079569, 1.1197608
9: -6.6679935, -5.0688710, -6.6729627, -5.0263948, -1.5786629, 1.5428772

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7393106
time: 4.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7512163
time: 4.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.1566496, -10.5546093, -1.7239423, 1.6931186
1: -11.3276834, -8.4670172, -11.2935276, -8.4476376, -2.0518928, 2.0044835
2: -10.7271967, -8.5144558, -10.6638432, -8.5587721, -1.7851784, 1.8081813
3: -4.4033794, -2.3413148, -4.4157672, -2.3489795, -1.6380935, 1.6401882
4: -15.1088705, -12.5990896, -15.1383734, -12.5446138, -1.8347807, 1.8061004
5: 8.2397890, 9.6893950, 8.2378788, 9.6845093, -1.0470369, 1.0680406
6: -4.7403202, -2.3233070, -4.7290587, -2.3376524, -1.6959114, 1.6925139
7: -15.7568035, -12.9566870, -15.7401867, -12.9516220, -2.2442598, 2.2202668
8: -0.7772584, 0.9293244, -0.7856748, 0.9181707, -1.1185837, 1.1394554
9: -6.6664100, -5.0656857, -6.6729627, -5.0263948, -1.5824790, 1.5529675

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7430712
time: 4.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
time: 4.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1579695, -10.5516567, -13.1857891, -10.5428429, -1.6759820, 1.7448914
1: -11.2414818, -8.4821720, -11.2851009, -8.4629269, -1.8697619, 1.9538674
2: -10.7234631, -8.5876637, -10.6992455, -8.5019779, -1.8624749, 1.7055187
3: -4.3974495, -2.3615463, -4.4218082, -2.3488288, -1.6121998, 1.6199524
4: -15.0964365, -12.5836086, -15.1325903, -12.5524931, -1.8106723, 1.7864127
5: 8.2681160, 9.6821699, 8.2256489, 9.6866417, -1.0239899, 1.0560641
6: -4.7094274, -2.3208578, -4.7589550, -2.3144078, -1.6563697, 1.7421355
7: -15.7122040, -12.9763803, -15.7447891, -12.9501724, -2.2033854, 2.2019892
8: -0.7385950, 0.8699014, -0.7951312, 0.9445491, -1.1198440, 1.1074896
9: -6.6707439, -5.0698318, -6.6712294, -5.0237017, -1.5929098, 1.5387697

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7362350
time: 4.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7499435
time: 4.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1620159, -10.5502062, -13.1862564, -10.5428429, -1.6804376, 1.7479279
1: -11.2943897, -8.4369335, -11.2963495, -8.4629269, -1.8003335, 2.0804877
2: -10.7318192, -8.5673981, -10.7000055, -8.5012970, -1.8917775, 1.7175937
3: -4.3982763, -2.3640292, -4.4219270, -2.3501844, -1.6303735, 1.6257644
4: -15.0920048, -12.5891705, -15.1294212, -12.5524120, -1.7981043, 1.8064361
5: 8.2522135, 9.6946011, 8.2254696, 9.6854649, -1.0543345, 1.0477569
6: -4.7113104, -2.3202448, -4.7590933, -2.3143644, -1.6593585, 1.7427256
7: -15.7421875, -12.9503937, -15.7464657, -12.9500923, -2.2076616, 2.2280831
8: -0.7728252, 0.9029632, -0.8030822, 0.9445934, -1.1022124, 1.1467431
9: -6.6725430, -5.0673070, -6.6712637, -5.0233111, -1.5921054, 1.5403380

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7359346
time: 5.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7459542
time: 5.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.1620617, -10.5540009, -1.6863098, 1.6742148
1: -11.2934866, -8.4585981, -11.3223324, -8.4282084, -1.9970312, 1.9693086
2: -10.6638260, -8.5663605, -10.7312403, -8.5680332, -1.7158775, 1.7976067
3: -4.4163847, -2.3490036, -4.3984842, -2.3431964, -1.6496124, 1.6322222
4: -15.1383724, -12.5146065, -15.1122246, -12.5858021, -1.8174019, 1.8646221
5: 8.2280121, 9.6844997, 8.2517548, 9.7013798, -1.0814095, 1.0448246
6: -4.7367697, -2.3376508, -4.7115936, -2.3240569, -1.6954656, 1.6477799
7: -15.7401762, -12.9545841, -15.7497559, -12.9487991, -2.2296820, 2.2363620
8: -0.7961533, 0.9181619, -0.7762470, 0.9030502, -1.1363223, 1.1251206
9: -6.6729279, -5.0165362, -6.6744304, -5.0671225, -1.5443482, 1.5911393

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511265, upper bound: 0.7421467
time: 3.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511265, upper bound: 0.7421467
time: 3.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.1627216, -10.5502033, -1.7512541, 1.6855881
1: -11.3031998, -8.4738846, -11.3232145, -8.4369335, -2.0834079, 1.9657521
2: -10.7027636, -8.5088758, -10.7405539, -8.5670967, -1.7344079, 1.9045813
3: -4.4226265, -2.3455589, -4.3988109, -2.3429549, -1.6564212, 1.6351194
4: -15.1351519, -12.5223141, -15.1122541, -12.5887508, -1.8111205, 1.8674645
5: 8.2154884, 9.6887102, 8.2516556, 9.7012825, -1.0906048, 1.0566378
6: -4.7669148, -2.3143487, -4.7118077, -2.3201573, -1.7544079, 1.6597624
7: -15.7503185, -12.9529018, -15.7504702, -12.9495525, -2.2322950, 2.2358680
8: -0.8150344, 0.9446130, -0.7797508, 0.9030869, -1.1622910, 1.1479175
9: -6.6713448, -5.0133581, -6.6731215, -5.0667696, -1.5449338, 1.6025105

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7499413, upper bound: 0.7386207
time: 6.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7459521, upper bound: 0.7386232
time: 4.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.1600590, -10.5419407, -1.6870208, 1.6618800
1: -11.2934866, -8.4585981, -11.2978497, -8.4350758, -2.0044241, 1.9681413
2: -10.6638260, -8.5663605, -10.7068071, -8.5625324, -1.6944978, 1.7466896
3: -4.4163847, -2.3490036, -4.4176960, -2.3474734, -1.6264205, 1.6255255
4: -15.1383724, -12.5146065, -15.1385098, -12.5090113, -1.8206153, 1.8183870
5: 8.2280121, 9.6844997, 8.2273998, 9.7007198, -1.0653543, 1.0543404
6: -4.7367697, -2.3376508, -4.7382407, -2.3151054, -1.6719170, 1.6423798
7: -15.7401762, -12.9545841, -15.7433491, -12.9449978, -2.2473612, 2.2416196
8: -0.7961533, 0.9181619, -0.8140235, 0.9183445, -1.1044211, 1.1191213
9: -6.6729279, -5.0165362, -6.6793642, -5.0147805, -1.5725975, 1.5780592

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511267, upper bound: 0.7392570
time: 4.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511267, upper bound: 0.7392569
time: 4.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.1607265, -10.5381422, -1.7519159, 1.6732535
1: -11.3031998, -8.4738846, -11.2987289, -8.4438009, -2.0907664, 1.9645851
2: -10.7027636, -8.5088758, -10.7161217, -8.5616302, -1.7130342, 1.8537393
3: -4.4226265, -2.3455589, -4.4180040, -2.3472333, -1.6332436, 1.6284130
4: -15.1351519, -12.5223141, -15.1385374, -12.5119610, -1.8143225, 1.8212509
5: 8.2154884, 9.6887102, 8.2272949, 9.7006359, -1.0745736, 1.0661325
6: -4.7669148, -2.3143487, -4.7384491, -2.3112073, -1.7308950, 1.6543493
7: -15.7503185, -12.9529018, -15.7440968, -12.9457512, -2.2498875, 2.2411370
8: -0.8150344, 0.9446130, -0.8175254, 0.9183803, -1.1304002, 1.1419163
9: -6.6713448, -5.0133581, -6.6780562, -5.0144253, -1.5731821, 1.5894399

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7500556, upper bound: 0.7369712
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7463142, upper bound: 0.7369689
time: 7.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.3020658, -10.5097313, -1.6954269, 1.7496004
1: -11.3179722, -8.4517279, -11.4193554, -8.4559507, -2.0017538, 2.1361785
2: -10.6882601, -8.5720196, -10.7133265, -8.5700626, -1.7307820, 1.7966866
3: -4.3971014, -2.3447456, -4.5674419, -2.3432286, -1.6291776, 1.6545014
4: -15.1120892, -12.5913963, -15.1517267, -12.4656668, -1.7415891, 1.8060503
5: 8.2523460, 9.6852093, 8.2329454, 9.6461153, -1.0438101, 1.1153684
6: -4.7101393, -2.3466036, -4.7792645, -2.3436372, -1.6436391, 1.6406455
7: -15.7467308, -12.9583788, -15.7843122, -12.9975567, -1.9629774, 2.3148580
8: -0.7583771, 0.9028707, -0.7592177, 0.9712560, -1.2417002, 1.1197314
9: -6.6679935, -5.0688710, -6.6707263, -4.9717278, -1.6362534, 1.1771541

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7281414
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7400543
time: 4.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.3020658, -10.5097313, -1.7513080, 1.7806609
1: -11.3276834, -8.4670172, -11.4193554, -8.4559507, -2.0751877, 2.1823146
2: -10.7271967, -8.5144558, -10.7133265, -8.5700626, -1.7888331, 1.8903208
3: -4.4033794, -2.3413148, -4.5674419, -2.3432286, -1.6356950, 1.6578209
4: -15.1088705, -12.5990896, -15.1517267, -12.4656668, -1.7463031, 1.7986331
5: 8.2397890, 9.6893950, 8.2329454, 9.6461153, -1.0540795, 1.1244961
6: -4.7403202, -2.3233070, -4.7792645, -2.3436372, -1.6930294, 1.6807253
7: -15.7568035, -12.9566870, -15.7843122, -12.9975567, -1.9729176, 2.3137937
8: -0.7772584, 0.9293244, -0.7592177, 0.9712560, -1.2523270, 1.1427641
9: -6.6664100, -5.0656857, -6.6707263, -4.9717278, -1.6400700, 1.1999867

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7319267
time: 4.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7438395
time: 3.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1579695, -10.5516567, -13.3314791, -10.4979630, -1.7033472, 1.8260829
1: -11.2414818, -8.4821720, -11.4110804, -8.4712267, -1.8930559, 2.1323450
2: -10.7234631, -8.5876637, -10.7486973, -8.5131187, -1.8664041, 1.7876687
3: -4.3974495, -2.3615463, -4.5736394, -2.3430793, -1.6096663, 1.6328852
4: -15.0964365, -12.5836086, -15.1459675, -12.4734554, -1.7128139, 1.7792211
5: 8.2681160, 9.6821699, 8.2206593, 9.6482716, -1.0274830, 1.1134506
6: -4.7094274, -2.3208578, -4.8097920, -2.3204281, -1.6534696, 1.7238548
7: -15.7122040, -12.9763803, -15.7891588, -12.9958897, -1.9233274, 2.2959113
8: -0.7385950, 0.8699014, -0.7686048, 0.9977300, -1.2537651, 1.1028748
9: -6.6707439, -5.0698318, -6.6690531, -4.9690161, -1.6506271, 1.1785181

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7253072
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7389715
time: 4.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1620159, -10.5502062, -13.3319216, -10.4979649, -1.7078037, 1.8301790
1: -11.2943897, -8.4369335, -11.4223089, -8.4712267, -1.8236279, 2.2589417
2: -10.7318192, -8.5673981, -10.7494612, -8.5124264, -1.8965588, 1.7997437
3: -4.3982763, -2.3640292, -4.5737267, -2.3444390, -1.6278548, 1.6339025
4: -15.0920048, -12.5891705, -15.1427879, -12.4733906, -1.7000365, 1.7992582
5: 8.2522135, 9.6946011, 8.2204857, 9.6470776, -1.0562639, 1.1051806
6: -4.7113104, -2.3202448, -4.8099198, -2.3203835, -1.6564598, 1.7251091
7: -15.7421875, -12.9503937, -15.7907906, -12.9958496, -1.9171829, 2.3219290
8: -0.7728252, 0.9029632, -0.7765419, 0.9977729, -1.2361336, 1.1450903
9: -6.6725430, -5.0673070, -6.6690569, -4.9686284, -1.6498156, 1.1800077

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7249201
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7350131
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.3077297, -10.5091209, -1.7136774, 1.7642162
1: -11.2934866, -8.4585981, -11.4483929, -8.4365149, -2.0203238, 2.1477473
2: -10.6638260, -8.5663605, -10.7807407, -8.5795660, -1.7200832, 1.8797338
3: -4.4163847, -2.3490036, -4.5501785, -2.3374286, -1.6474643, 1.6460018
4: -15.1383724, -12.5146065, -15.1255865, -12.5068340, -1.7233491, 1.8571463
5: 8.2280121, 9.6844997, 8.2468157, 9.6629982, -1.0866466, 1.1012421
6: -4.7367697, -2.3376508, -4.7616272, -2.3300266, -1.6925845, 1.6353366
7: -15.7401762, -12.9545841, -15.7944136, -12.9947395, -1.9547210, 2.3307400
8: -0.7961533, 0.9181619, -0.7498717, 0.9561501, -1.2700591, 1.1278701
9: -6.6729279, -5.0165362, -6.6723170, -5.0124073, -1.6020932, 1.2308540

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7315535, upper bound: 0.7310116
time: 3.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7315535, upper bound: 0.7310117
time: 3.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.3084154, -10.5053253, -1.7786217, 1.7738574
1: -11.3031998, -8.4738846, -11.4493256, -8.4452410, -2.1067004, 2.1443193
2: -10.7027636, -8.5088758, -10.7900562, -8.5786839, -1.7388115, 1.9867058
3: -4.4226265, -2.3455589, -4.5505123, -2.3371875, -1.6542840, 1.6493871
4: -15.1351519, -12.5223141, -15.1256161, -12.5097771, -1.7270799, 1.8599668
5: 8.2154884, 9.6887102, 8.2467136, 9.6629019, -1.1012385, 1.1130631
6: -4.7669148, -2.3143487, -4.7618217, -2.3261244, -1.7515273, 1.6498904
7: -15.7503185, -12.9529018, -15.7952347, -12.9955006, -1.9647894, 2.3304305
8: -0.8150344, 0.9446130, -0.7533553, 0.9561877, -1.2960286, 1.1558046
9: -6.6713448, -5.0133581, -6.6710076, -5.0120454, -1.6027160, 1.2561219

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7305582, upper bound: 0.7276286
time: 3.54 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7266000, upper bound: 0.7276285
time: 3.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.3056011, -10.4970589, -1.7143860, 1.7443078
1: -11.2934866, -8.4585981, -11.4239092, -8.4433861, -2.0277176, 2.1465826
2: -10.6638260, -8.5663605, -10.7563124, -8.5739956, -1.6988525, 1.8288169
3: -4.4163847, -2.3490036, -4.5694919, -2.3417053, -1.6242666, 1.6459863
4: -15.1383724, -12.5146065, -15.1518612, -12.4299374, -1.7393913, 1.8109112
5: 8.2280121, 9.6844997, 8.2223921, 9.6623058, -1.0653965, 1.1107609
6: -4.7367697, -2.3376508, -4.7882972, -2.3210754, -1.6690354, 1.6339109
7: -15.7401762, -12.9545841, -15.7879543, -12.9910526, -1.9783030, 2.3359981
8: -0.7961533, 0.9181619, -0.7876420, 0.9714403, -1.2381582, 1.1255219
9: -6.6729279, -5.0165362, -6.6771307, -4.9600654, -1.6303425, 1.1709392

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7280845
time: 3.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7280845
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.3062897, -10.4932652, -1.7792821, 1.7539489
1: -11.3031998, -8.4738846, -11.4248438, -8.4521084, -2.1140594, 2.1431541
2: -10.7027636, -8.5088758, -10.7656260, -8.5731544, -1.7175875, 1.9358637
3: -4.4226265, -2.3455589, -4.5698099, -2.3414660, -1.6311026, 1.6493700
4: -15.1351519, -12.5223141, -15.1518917, -12.4328842, -1.7431388, 1.8137527
5: 8.2154884, 9.6887102, 8.2222862, 9.6622219, -1.0799601, 1.1225619
6: -4.7669148, -2.3143487, -4.7884893, -2.3171744, -1.7280135, 1.6486142
7: -15.7503185, -12.9529018, -15.7888126, -12.9918098, -1.9883356, 2.3356996
8: -0.8150344, 0.9446130, -0.7911248, 0.9714806, -1.2641375, 1.1534672
9: -6.6713448, -5.0133581, -6.6758208, -4.9597025, -1.6309624, 1.1962125

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7306805, upper bound: 0.7259603
time: 4.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7269178, upper bound: 0.7259603
time: 3.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.3041668, -10.5235815, -13.1620617, -10.5540009, -1.7640271, 1.6859374
1: -11.4438000, -8.4600401, -11.3223324, -8.4282084, -2.1917620, 2.0009239
2: -10.7377462, -8.5833111, -10.7312403, -8.5680332, -1.7798429, 1.7537231
3: -4.5487471, -2.3389950, -4.3984842, -2.3431964, -1.6364796, 1.6143947
4: -15.1254501, -12.5124207, -15.1122246, -12.5858021, -1.7895403, 1.7013259
5: 8.2474213, 9.6468496, 8.2517548, 9.7013798, -1.1079030, 1.0397897
6: -4.7603240, -2.3525896, -4.7115936, -2.3240569, -1.6541424, 1.6311355
7: -15.7909136, -13.0042801, -15.7497559, -12.9487991, -2.3332567, 1.9599104
8: -0.7319293, 0.9559593, -0.7762470, 0.9030502, -1.0947604, 1.2412779
9: -6.6658897, -5.0142002, -6.6744304, -5.0671225, -1.1515059, 1.6167850

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226006
time: 4.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226006
time: 3.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.3020535, -10.5115213, -13.1620617, -10.5540009, -1.7738781, 1.7015808
1: -11.4193163, -8.4669085, -11.3223324, -8.4282084, -2.1748662, 1.9926035
2: -10.7133112, -8.5775843, -10.7312403, -8.5680332, -1.7980165, 1.8006163
3: -4.5681343, -2.3432529, -4.3984842, -2.3431964, -1.6610920, 1.6298251
4: -15.1517258, -12.4355240, -15.1122246, -12.5858021, -1.8099356, 1.7639999
5: 8.2230186, 9.6461086, 8.2517548, 9.7013798, -1.1378851, 1.0473304
6: -4.7869792, -2.3436360, -4.7115936, -2.3240569, -1.6814561, 1.6448984
7: -15.7843018, -13.0005989, -15.7497559, -12.9487991, -2.3232069, 1.9601469
8: -0.7696996, 0.9712491, -0.7762470, 0.9030502, -1.1344762, 1.2588642
9: -6.6707015, -4.9618669, -6.6744304, -5.0671225, -1.1784110, 1.6487312

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226005
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226006
time: 3.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.3041668, -10.5235815, -13.3077297, -10.5091209, -1.6782217, 1.6569047
1: -11.4438000, -8.4600401, -11.4483929, -8.4365149, -2.0507631, 2.0144954
2: -10.7377462, -8.5833111, -10.7807407, -8.5795660, -1.6969433, 1.7490237
3: -4.5487471, -2.3389950, -4.5501785, -2.3374286, -1.4417737, 1.4412878
4: -15.1254501, -12.5124207, -15.1255865, -12.5068340, -1.6333592, 1.6278450
5: 8.2474213, 9.6468496, 8.2468157, 9.6629982, -1.0214266, 1.0088539
6: -4.7603240, -2.3525896, -4.7616272, -2.3300266, -1.5715966, 1.5425972
7: -15.7909136, -13.0042801, -15.7944136, -12.9947395, -1.9752865, 1.9665864
8: -0.7319293, 0.9559593, -0.7498717, 0.9561501, -1.0634346, 1.0795963
9: -6.6658897, -5.0142002, -6.6723170, -5.0124073, -1.1815495, 1.1910598

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226005
time: 4.31 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226006
time: 3.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.3020535, -10.5115213, -13.3077297, -10.5091209, -1.6880703, 1.6781855
1: -11.4193163, -8.4669085, -11.4483929, -8.4365149, -2.0338631, 2.0061772
2: -10.7133112, -8.5775843, -10.7807407, -8.5795660, -1.7150521, 1.7959158
3: -4.5681343, -2.3432529, -4.5501785, -2.3374286, -1.4663851, 1.4510393
4: -15.1517258, -12.4355240, -15.1255865, -12.5068340, -1.6496670, 1.6905172
5: 8.2230186, 9.6461086, 8.2468157, 9.6629982, -1.0558397, 1.0163950
6: -4.7869792, -2.3436360, -4.7616272, -2.3300266, -1.5989089, 1.5529082
7: -15.7843018, -13.0005989, -15.7944136, -12.9947395, -1.9614296, 1.9668248
8: -0.7696996, 0.9712491, -0.7498717, 0.9561501, -1.1031578, 1.0961938
9: -6.6707015, -4.9618669, -6.6723170, -5.0124073, -1.2084546, 1.2609122

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226006
time: 4.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226006
time: 3.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3339596, -10.5118170, -13.1627216, -10.5502033, -1.8241816, 1.6973114
1: -11.4536533, -8.4753218, -11.3232145, -8.4369335, -2.2787104, 1.9973679
2: -10.7766523, -8.5255909, -10.7405539, -8.5670967, -1.7983899, 1.8610916
3: -4.5550556, -2.3355613, -4.3988109, -2.3429549, -1.6420894, 1.6171722
4: -15.1222477, -12.5201187, -15.1122541, -12.5887508, -1.7835388, 1.7036774
5: 8.2348223, 9.6510172, 8.2516556, 9.7012825, -1.1180515, 1.0511601
6: -4.7910752, -2.3293247, -4.7118077, -2.3201573, -1.7093410, 1.6430864
7: -15.8011827, -13.0025578, -15.7504702, -12.9495525, -2.3360987, 1.9644229
8: -0.7507010, 0.9825034, -0.7797508, 0.9030869, -1.1194539, 1.2642512
9: -6.6642218, -5.0110049, -6.6731215, -5.0667696, -1.1551638, 1.6282806

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7253052, upper bound: 0.7192174
time: 3.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7249181, upper bound: 0.7192174
time: 3.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3320036, -10.4997549, -13.1627216, -10.5502033, -1.8339930, 1.7129550
1: -11.4291677, -8.4821873, -11.3232145, -8.4369335, -2.2618480, 1.9890461
2: -10.7522211, -8.5199451, -10.7405539, -8.5670967, -1.8165569, 1.9078970
3: -4.5744066, -2.3398066, -4.3988109, -2.3429549, -1.6666737, 1.6326127
4: -15.1485214, -12.4432383, -15.1122541, -12.5887508, -1.8039455, 1.7663364
5: 8.2104530, 9.6503019, 8.2516556, 9.7012825, -1.1480052, 1.0587527
6: -4.8176918, -2.3203642, -4.7118077, -2.3201573, -1.7365580, 1.6568639
7: -15.7946510, -12.9988842, -15.7504702, -12.9495525, -2.3261442, 1.9646714
8: -0.7884698, 0.9977911, -0.7797508, 0.9030869, -1.1591148, 1.2818387
9: -6.6690321, -4.9586730, -6.6731215, -5.0667696, -1.1820693, 1.6602154

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7253052, upper bound: 0.7192174
time: 3.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7249181, upper bound: 0.7192174
time: 3.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.07 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7393106
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7512163
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7430712
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7362350
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7499435
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7359346
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7386213, upper bound: 0.7459542
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7511265, upper bound: 0.7421467
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7511265, upper bound: 0.7421467
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7499413, upper bound: 0.7386207
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7459521, upper bound: 0.7386232
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7511267, upper bound: 0.7392570
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7511267, upper bound: 0.7392569
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7500556, upper bound: 0.7369712
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7463142, upper bound: 0.7369689
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7281414
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7400543
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7319267
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7438395
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7253072
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7389715
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7249201
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7350131
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7315535, upper bound: 0.7310116
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7315535, upper bound: 0.7310117
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7305582, upper bound: 0.7276286
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7266000, upper bound: 0.7276285
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7280845
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7280845
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7306805, upper bound: 0.7259603
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7269178, upper bound: 0.7259603
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226006
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226006
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226005
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7281393, upper bound: 0.7226006
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226005
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226006
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226006
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7197283, upper bound: 0.7226006
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7253052, upper bound: 0.7192174
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7249181, upper bound: 0.7192174
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7253052, upper bound: 0.7192174
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.07
Output dim: 5, lower bound: -0.7249181, upper bound: 0.7192174
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280819
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7280820
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7315554
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7225988, upper bound: 0.7354263
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7305603
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7192158, upper bound: 0.7266021
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7196712
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7315536, upper bound: 0.7235437
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7269176, upper bound: 0.7172499
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.07
Output dim: 5, lower bound: -0.7269176, upper bound: 0.7175494
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.139000654220581
rel_dist={5: [-0.78768892317715, 0.7876895182364123]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6815778
time: 4.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749160
time: 3.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.82
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6815778
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.82
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749160

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.1627960, -10.4737358, -1.6383018, 1.6017058
1: -11.2995148, -8.4136467, -11.3003588, -8.3620749, -1.9189978, 1.8883810
2: -10.7255239, -8.5416927, -10.7255354, -8.5209312, -1.7672639, 1.7447312
3: -4.4290442, -2.3468359, -4.4305668, -2.2836435, -1.6021833, 1.5795250
4: -15.1385698, -12.5005589, -15.1751680, -12.4989319, -1.7643862, 1.7776523
5: 8.2217073, 9.7064924, 8.2196198, 9.7211447, -1.0603046, 1.0434932
6: -4.7417126, -2.3063393, -4.7435021, -2.2806067, -1.6394768, 1.6160386
7: -15.7444887, -12.9351664, -15.7447205, -12.9068966, -2.1686831, 2.1333294
8: -0.8229923, 0.9184313, -0.8494754, 0.9187570, -1.1096857, 1.1579499
9: -6.6816912, -5.0027695, -6.7047820, -5.0025816, -1.5322986, 1.5543733

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693374
time: 5.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6760374
time: 4.13 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -13.3069639, -10.4817972, -13.1606512, -10.4731188, -1.7398300, 1.6490161
1: -11.4256668, -8.4219580, -11.2988787, -8.3831825, -2.0960016, 1.9176061
2: -10.7750311, -8.5532455, -10.7254887, -8.5322371, -1.8436217, 1.7451644
3: -4.5807705, -2.3410683, -4.4288559, -2.2959676, -1.6259804, 1.5890744
4: -15.1519251, -12.4215012, -15.1697922, -12.5002794, -1.7649174, 1.6838064
5: 8.2167139, 9.6680660, 8.2220287, 9.7037144, -1.1187311, 1.0242308
6: -4.7917042, -2.3123016, -4.7406893, -2.2900782, -1.6167114, 1.6301465
7: -15.7892628, -12.9812613, -15.7442532, -12.9337959, -2.2520809, 1.8411140
8: -0.7966328, 0.9715328, -0.8277798, 0.9180880, -1.0769272, 1.3004463
9: -6.6794314, -4.9480395, -6.6974730, -5.0028448, -1.2155576, 1.6082973

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6626756
time: 3.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693756
time: 3.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693374
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6760374
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6626756
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693756

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.1613684, -10.5349464, -13.1647110, -10.4942093, -1.6181517, 1.5899317
1: -11.2994976, -8.4160891, -11.3247900, -8.3681211, -1.9140205, 1.9031174
2: -10.7255192, -8.5550365, -10.7499447, -8.5458441, -1.7159410, 1.7014511
3: -4.4147496, -2.3468454, -4.4005256, -2.2793865, -1.5669727, 1.5431554
4: -15.1385670, -12.5457249, -15.1488791, -12.5821047, -1.6887941, 1.7194908
5: 8.2406349, 9.7064877, 8.2494383, 9.7217617, -1.0374290, 1.0064920
6: -4.7284250, -2.3063426, -4.7139254, -2.2895546, -1.6104431, 1.5834227
7: -15.7444878, -12.9394283, -15.7510538, -12.9171677, -2.1553478, 2.1375446
8: -0.8066788, 0.9184277, -0.8091183, 0.9034505, -1.0739286, 1.1139858
9: -6.6816726, -5.0292954, -6.6997247, -5.0663037, -1.4888372, 1.5434518

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570860
time: 6.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622635
time: 5.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.1613798, -10.5291605, -13.1627598, -10.4821548, -1.6182995, 1.6000631
1: -11.2995005, -8.4167728, -11.3003035, -8.3749886, -1.9232793, 1.8851237
2: -10.7255211, -8.5464029, -10.7255182, -8.5404100, -1.6951823, 1.7436910
3: -4.4255018, -2.3468423, -4.4196863, -2.2836750, -1.5999994, 1.5346749
4: -15.1385698, -12.5021210, -15.1751633, -12.5053921, -1.6872649, 1.7736449
5: 8.2232246, 9.7064896, 8.2250690, 9.7211342, -1.0571946, 1.0131735
6: -4.7410054, -2.3063388, -4.7405791, -2.2806065, -1.6392632, 1.5763416
7: -15.7444859, -12.9367161, -15.7447109, -12.9133053, -2.1711855, 2.1303830
8: -0.8223681, 0.9184284, -0.8468938, 0.9187460, -1.1095438, 1.1056492
9: -6.6816797, -5.0055199, -6.7047305, -5.0139589, -1.5132947, 1.5513763

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637877
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689649
time: 3.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -13.3069506, -10.4900675, -13.1626348, -10.4935932, -1.7157879, 1.6373279
1: -11.4256477, -8.4243975, -11.3233109, -8.3892307, -2.0910206, 1.9323447
2: -10.7750263, -8.5666552, -10.7498980, -8.5571766, -1.7922668, 1.7024646
3: -4.5664873, -2.3410795, -4.3987818, -2.2917123, -1.5983028, 1.5527442
4: -15.1519222, -12.4667854, -15.1435070, -12.5835285, -1.6893439, 1.6301117
5: 8.2356806, 9.6680660, 8.2518597, 9.7043486, -1.0958838, 0.9857317
6: -4.7784252, -2.3123050, -4.7111049, -2.2990298, -1.5919027, 1.5975213
7: -15.7892590, -12.9854088, -15.7506065, -12.9440022, -2.2387428, 1.8509274
8: -0.7803226, 0.9715278, -0.7874227, 0.9027820, -1.0433059, 1.2564969
9: -6.6794271, -4.9745660, -6.6924930, -5.0665669, -1.1454163, 1.5974512

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6504259
time: 3.74 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6556029
time: 3.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -13.3069620, -10.4842815, -13.1606197, -10.4815350, -1.7069664, 1.6473722
1: -11.4256506, -8.4250803, -11.2988253, -8.3960981, -2.1002994, 1.9143472
2: -10.7750282, -8.5579576, -10.7254715, -8.5517139, -1.7715092, 1.7441292
3: -4.5772333, -2.3410764, -4.4179764, -2.2959988, -1.6234245, 1.5443072
4: -15.1519241, -12.4230604, -15.1697893, -12.5067368, -1.6878352, 1.6798680
5: 8.2182121, 9.6680641, 8.2274704, 9.7037029, -1.1156205, 0.9885231
6: -4.7909989, -2.3123019, -4.7377653, -2.2900791, -1.6164725, 1.5904408
7: -15.7892609, -12.9828033, -15.7442417, -12.9402046, -2.2546492, 1.8393824
8: -0.7960086, 0.9715285, -0.8251977, 0.9180758, -1.0764892, 1.2481604
9: -6.6794248, -4.9507914, -6.6974230, -5.0142212, -1.1264520, 1.6052999

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6571217
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6623030
time: 3.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570860
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622635
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637877
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689649
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6504259
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6556029
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6571217
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.69
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6623030

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1566467, -10.5562611, -13.1628866, -10.5033951, -1.5991979, 1.5494711
1: -11.2935266, -8.4481239, -11.3225317, -8.3800697, -1.8867006, 1.8458705
2: -10.6638412, -8.5604353, -10.7237740, -8.5477180, -1.6416769, 1.6717019
3: -4.4129162, -2.3489814, -4.3998394, -2.2802629, -1.5650425, 1.5405145
4: -15.1383734, -12.5532913, -15.1487961, -12.5848970, -1.6835303, 1.7125392
5: 8.2414856, 9.6845083, 8.2497749, 9.7137089, -1.0234435, 0.9832200
6: -4.7264080, -2.3376534, -4.7131648, -2.3018637, -1.5932789, 1.5409734
7: -15.7401829, -12.9524250, -15.7495527, -12.9219561, -2.1470537, 2.1246238
8: -0.7824209, 0.9181695, -0.8001573, 0.9033444, -1.0475948, 1.0994103
9: -6.6729603, -5.0316868, -6.6965303, -5.0671878, -1.4777803, 1.5367274

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570877
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570858
time: 6.89 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1863976, -10.5444908, -13.1638756, -10.4981403, -1.6679621, 1.5608106
1: -11.3032370, -8.4634132, -11.3238392, -8.3905287, -1.9786224, 1.8434556
2: -10.7027769, -8.5028753, -10.7377510, -8.5464306, -1.6590574, 1.7850156
3: -4.4191785, -2.3455374, -4.4003115, -2.2798851, -1.5720253, 1.5434971
4: -15.1351528, -12.5610037, -15.1488419, -12.5885220, -1.6767197, 1.7161257
5: 8.2289534, 9.6887178, 8.2496204, 9.7141743, -1.0340695, 0.9948205
6: -4.7565508, -2.3143501, -4.7134972, -2.2958107, -1.6562486, 1.5510690
7: -15.7503300, -12.9507360, -15.7505817, -12.9225864, -2.1495152, 2.1244211
8: -0.8013029, 0.9446208, -0.8057313, 0.9034016, -1.0712509, 1.1241865
9: -6.6713777, -5.0285101, -6.6950650, -5.0666652, -1.4784393, 1.5488424

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622648
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622627
time: 5.49 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1566563, -10.5504742, -13.1609497, -10.4913397, -1.5992327, 1.5596461
1: -11.2935266, -8.4488049, -11.2980490, -8.3869362, -1.8959837, 1.8278675
2: -10.6638441, -8.5515947, -10.6993446, -8.5422087, -1.6209257, 1.7139554
3: -4.4236708, -2.3489780, -4.4190259, -2.2845409, -1.5980759, 1.5320168
4: -15.1383743, -12.5097094, -15.1750803, -12.5081854, -1.6819911, 1.7666974
5: 8.2240858, 9.6845102, 8.2254162, 9.7130547, -1.0431826, 0.9899597
6: -4.7389874, -2.3376493, -4.7398109, -2.2929125, -1.6221023, 1.5338809
7: -15.7401829, -12.9497242, -15.7431383, -12.9180965, -2.1631432, 2.1173701
8: -0.7981102, 0.9181702, -0.8379345, 0.9186392, -1.0832105, 1.0910747
9: -6.6729646, -5.0079112, -6.7015352, -5.0148449, -1.5022383, 1.5446377

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637877
time: 4.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637877
time: 3.93 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1864090, -10.5387077, -13.1619434, -10.4860859, -1.6679473, 1.5709660
1: -11.3032408, -8.4640923, -11.2993555, -8.3973961, -1.9878721, 1.8254628
2: -10.7027817, -8.4941101, -10.7133207, -8.5409737, -1.6383114, 1.8272645
3: -4.4299331, -2.3455355, -4.4194803, -2.2841644, -1.6050568, 1.5349877
4: -15.1351528, -12.5174198, -15.1751242, -12.5118122, -1.6751685, 1.7702832
5: 8.2115650, 9.6887207, 8.2252569, 9.7135363, -1.0538170, 1.0015383
6: -4.7691312, -2.3143466, -4.7401395, -2.2868621, -1.6850710, 1.5439634
7: -15.7503271, -12.9480457, -15.7442169, -12.9187288, -2.1655140, 2.1172318
8: -0.8169909, 0.9446225, -0.8435040, 0.9186945, -1.1068676, 1.1158489
9: -6.6713853, -5.0047336, -6.7000723, -5.0143213, -1.5028987, 1.5567422

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689649
time: 4.14 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689649
time: 4.05 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3020611, -10.5113811, -13.1607990, -10.5027752, -1.6966333, 1.5968776
1: -11.4193535, -8.4564342, -11.3210440, -8.4011793, -2.0629277, 1.8751051
2: -10.7133245, -8.5717316, -10.7237272, -8.5590534, -1.7180238, 1.6726792
3: -4.5645933, -2.3432310, -4.3980923, -2.2925713, -1.5961027, 1.5500658
4: -15.1517258, -12.4743414, -15.1434269, -12.5863237, -1.6841388, 1.6188891
5: 8.2365532, 9.6461134, 8.2521992, 9.6962967, -1.0818893, 0.9600767
6: -4.7766171, -2.3436375, -4.7103405, -2.3113389, -1.5749002, 1.5550737
7: -15.7843122, -12.9983559, -15.7490950, -12.9487877, -2.2294579, 1.8373249
8: -0.7559652, 0.9712560, -0.7784333, 0.9026768, -1.0154946, 1.2419024
9: -6.6707249, -4.9770184, -6.6892991, -5.0674548, -1.1301453, 1.5905156

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6504259
time: 3.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6504260
time: 4.10 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3320122, -10.4996128, -13.1617928, -10.4975185, -1.7605948, 1.6082106
1: -11.4292049, -8.4717159, -11.3223581, -8.4116383, -2.1554527, 1.8726850
2: -10.7522335, -8.5140104, -10.7377062, -8.5577641, -1.7354116, 1.7860742
3: -4.5708861, -2.3397861, -4.3985662, -2.2921975, -1.6017833, 1.5530307
4: -15.1485224, -12.4820595, -15.1434689, -12.5899467, -1.6775880, 1.6220367
5: 8.2239780, 9.6503048, 8.2520428, 9.6967640, -1.0934472, 0.9713664
6: -4.8073292, -2.3203678, -4.7106748, -2.3052828, -1.6341789, 1.5651615
7: -15.7946587, -12.9966345, -15.7501345, -12.9494181, -2.2321749, 1.8418009
8: -0.7747347, 0.9977977, -0.7839849, 0.9027314, -1.0381448, 1.2668583
9: -6.6690583, -4.9738240, -6.6878352, -5.0669289, -1.1340470, 1.6027441

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6556030
time: 4.45 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6556030
time: 4.02 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3020706, -10.5055943, -13.1587963, -10.4907169, -1.6878791, 1.6069667
1: -11.4193573, -8.4571171, -11.2965584, -8.4080477, -2.0722318, 1.8570988
2: -10.7133274, -8.5628185, -10.6992970, -8.5535145, -1.6972728, 1.7143598
3: -4.5753388, -2.3432283, -4.4173222, -2.2968488, -1.6212244, 1.5416110
4: -15.1517277, -12.4306383, -15.1697102, -12.5095310, -1.6826205, 1.6686401
5: 8.2190933, 9.6461124, 8.2278194, 9.6956282, -1.1015986, 0.9630075
6: -4.7891903, -2.3436341, -4.7369947, -2.3023868, -1.5993702, 1.5479796
7: -15.7843113, -12.9957619, -15.7426586, -12.9449930, -2.2456150, 1.8257287
8: -0.7716520, 0.9712572, -0.8162081, 0.9179697, -1.0486705, 1.2335672
9: -6.6707234, -4.9532433, -6.6942306, -5.0151129, -1.1111803, 1.5983505

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6571210
time: 6.56 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6571216
time: 5.46 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.3320208, -10.4938269, -13.1597958, -10.4854622, -1.7518816, 1.6182802
1: -11.4292088, -8.4723969, -11.2978764, -8.4185047, -2.1647220, 1.8546884
2: -10.7522354, -8.5051794, -10.7132750, -8.5522785, -1.7146664, 1.8277483
3: -4.5816317, -2.3397834, -4.4177718, -2.2964773, -1.6269064, 1.5445657
4: -15.1485252, -12.4383535, -15.1697512, -12.5131569, -1.6760583, 1.6717939
5: 8.2065287, 9.6503048, 8.2276583, 9.6961098, -1.1131659, 0.9742463
6: -4.8199048, -2.3203635, -4.7373257, -2.2963347, -1.6586637, 1.5580549
7: -15.7946606, -12.9940453, -15.7437449, -12.9456224, -2.2482386, 1.8302124
8: -0.7904239, 0.9978006, -0.8217599, 0.9180260, -1.0713210, 1.2585201
9: -6.6690540, -4.9500480, -6.6927643, -5.0145869, -1.1150813, 1.6105695

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6623010
time: 5.14 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6623030
time: 4.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.63 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570877
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570858
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622648
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622627
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637877
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637877
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689649
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689649
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6504259
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6504260
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6556030
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6556030
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6571210
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6571216
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6623010
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.63
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6623030

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1566467, -10.5562611, -13.1615391, -10.5563307, -1.5615640, 1.5485258
1: -11.2935266, -8.4481239, -11.3216839, -8.4316435, -1.8552704, 1.8450639
2: -10.6638412, -8.5604353, -10.7237625, -8.5686007, -1.6182928, 1.6708789
3: -4.4129162, -2.3489814, -4.3982882, -2.3434389, -1.5418968, 1.5400200
4: -15.1383734, -12.5532913, -15.1122036, -12.5866070, -1.6822534, 1.6979525
5: 8.2414856, 9.6845083, 8.2518520, 9.6990671, -1.0012693, 0.9779398
6: -4.7264080, -2.3376534, -4.7113757, -2.3275948, -1.5683570, 1.5394897
7: -15.7401829, -12.9524250, -15.7493277, -12.9501801, -2.1050787, 2.1179843
8: -0.7824209, 0.9181695, -0.7736735, 0.9030194, -1.0469103, 1.0506201
9: -6.6729603, -5.0316868, -6.6735115, -5.0673761, -1.4726868, 1.5096364

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6473993
time: 4.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6493521
time: 3.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1566467, -10.5562611, -13.3071823, -10.5114536, -1.5889306, 1.6292205
1: -11.2935266, -8.4481239, -11.4477100, -8.4399529, -1.8785634, 2.0234208
2: -10.6638412, -8.5604353, -10.7732630, -8.5800972, -1.6224113, 1.7530088
3: -4.4129162, -2.3489814, -4.5499749, -2.3376729, -1.5397172, 1.5491314
4: -15.1383734, -12.5532913, -15.1255608, -12.5076370, -1.5836763, 1.6904843
5: 8.2414856, 9.6845083, 8.2469168, 9.6606894, -1.0023434, 1.0343513
6: -4.7264080, -2.3376534, -4.7614303, -2.3335683, -1.5654755, 1.5204980
7: -15.7401829, -12.9524250, -15.7939177, -12.9961119, -1.8263931, 2.2122507
8: -0.7824209, 0.9181695, -0.7472873, 0.9561176, -1.1806483, 1.0538709
9: -6.6729603, -5.0316868, -6.6714001, -5.0126677, -1.5304098, 1.1205466

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6473992
time: 3.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6493515
time: 4.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1863976, -10.5444908, -13.1625299, -10.5510769, -1.6303287, 1.5598547
1: -11.3032370, -8.4634132, -11.3229933, -8.4421015, -1.9471922, 1.8426452
2: -10.7027769, -8.5028753, -10.7377396, -8.5672417, -1.6356840, 1.7841959
3: -4.4191785, -2.3455374, -4.3987608, -2.3430641, -1.5488720, 1.5430021
4: -15.1351528, -12.5610037, -15.1122456, -12.5902300, -1.6754336, 1.7015419
5: 8.2289534, 9.6887178, 8.2516966, 9.6995344, -1.0119468, 0.9895391
6: -4.7565508, -2.3143501, -4.7117066, -2.3215444, -1.6313267, 1.5495863
7: -15.7503300, -12.9507360, -15.7503586, -12.9508038, -2.1074886, 2.1177721
8: -0.8013029, 0.9446208, -0.7791660, 0.9030747, -1.0705636, 1.0753777
9: -6.6713777, -5.0285101, -6.6720467, -5.0668530, -1.4733491, 1.5217514

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6525281
time: 4.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6545074
time: 3.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1863976, -10.5444908, -13.3082132, -10.5061989, -1.6576967, 1.6383882
1: -11.3032370, -8.4634132, -11.4490948, -8.4504070, -1.9704838, 2.0211978
2: -10.7027769, -8.5028753, -10.7872429, -8.5788212, -1.6399970, 1.8663216
3: -4.4191785, -2.3455374, -4.5504627, -2.3372984, -1.5467162, 1.5526199
4: -15.1351528, -12.5610037, -15.1256056, -12.5112572, -1.5879564, 1.6940465
5: 8.2289534, 9.6887178, 8.2467575, 9.6611557, -1.0185099, 1.0459628
6: -4.7565508, -2.3143501, -4.7617340, -2.3275123, -1.6284456, 1.5326226
7: -15.7503300, -12.9507360, -15.7951097, -12.9967470, -1.8366585, 2.2123070
8: -0.8013029, 0.9446208, -0.7527628, 0.9561768, -1.2042990, 1.0839012
9: -6.6713777, -5.0285101, -6.6699343, -5.0121303, -1.5311251, 1.1469378

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6525281
time: 4.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6545074
time: 3.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1566563, -10.5504742, -13.1595364, -10.5442724, -1.5615993, 1.5586154
1: -11.2935266, -8.4488049, -11.2972002, -8.4385118, -1.8645539, 1.8270588
2: -10.6638441, -8.5515947, -10.6993294, -8.5630732, -1.5975411, 1.7131329
3: -4.4236708, -2.3489780, -4.4175091, -2.3477132, -1.5749273, 1.5315223
4: -15.1383743, -12.5097094, -15.1384869, -12.5098171, -1.6807151, 1.7521067
5: 8.2240858, 9.6845102, 8.2275000, 9.6984005, -1.0209951, 0.9846792
6: -4.7389874, -2.3376493, -4.7380171, -2.3186455, -1.5971832, 1.5323982
7: -15.7401829, -12.9497242, -15.7429028, -12.9463787, -2.1211672, 2.1107173
8: -0.7981102, 0.9181702, -0.8114500, 0.9183147, -1.0825260, 1.0422891
9: -6.6729646, -5.0079112, -6.6784458, -5.0150347, -1.4971442, 1.5174737

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6637877
time: 4.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6544458
time: 4.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1566563, -10.5504742, -13.3050594, -10.4993925, -1.5889654, 1.6510096
1: -11.2935266, -8.4488049, -11.4232244, -8.4468231, -1.8878474, 2.0054040
2: -10.6638441, -8.5515947, -10.7488317, -8.5745049, -1.6017919, 1.7952614
3: -4.4236708, -2.3489780, -4.5692968, -2.3419483, -1.5726981, 1.5471513
4: -15.1383743, -12.5097094, -15.1518402, -12.4307404, -1.5932360, 1.7446170
5: 8.2240858, 9.6845102, 8.2224941, 9.6599913, -1.0274992, 1.0410953
6: -4.7389874, -2.3376493, -4.7880993, -2.3246164, -1.5943041, 1.5172927
7: -15.7401829, -12.9497242, -15.7874365, -12.9924278, -1.8480124, 2.2049174
8: -0.7981102, 0.9181702, -0.7850575, 0.9714088, -1.2162633, 1.0486672
9: -6.6729646, -5.0079112, -6.6762142, -4.9603271, -1.5548677, 1.1821642

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6637876
time: 4.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6544458
time: 4.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1864090, -10.5387077, -13.1605358, -10.5390167, -1.6303144, 1.5699253
1: -11.3032408, -8.4640923, -11.2985115, -8.4489708, -1.9564419, 1.8246493
2: -10.7027817, -8.4941101, -10.7133093, -8.5617695, -1.6149385, 1.8264449
3: -4.4299331, -2.3455355, -4.4179573, -2.3473399, -1.5818996, 1.5344937
4: -15.1351528, -12.5174198, -15.1385279, -12.5134401, -1.6738834, 1.7556953
5: 8.2115650, 9.6887207, 8.2273388, 9.6988859, -1.0316812, 0.9962574
6: -4.7691312, -2.3143466, -4.7383471, -2.3125937, -1.6601505, 1.5424821
7: -15.7503271, -12.9480457, -15.7439833, -12.9470024, -2.1234884, 2.1105700
8: -0.8169909, 0.9446225, -0.8169415, 0.9183688, -1.1061800, 1.0670466
9: -6.6713853, -5.0047336, -6.6769814, -5.0145102, -1.4978085, 1.5295787

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6689646
time: 4.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6596230
time: 4.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.1864090, -10.5387077, -13.3060932, -10.4941368, -1.6576815, 1.6601996
1: -11.3032408, -8.4640923, -11.4246092, -8.4572773, -1.9797339, 2.0031888
2: -10.7027817, -8.4941101, -10.7628136, -8.5732822, -1.6193852, 1.9085693
3: -4.4299331, -2.3455355, -4.5697613, -2.3415761, -1.5796938, 1.5506370
4: -15.1351528, -12.5174198, -15.1518841, -12.4343634, -1.5975323, 1.7481766
5: 8.2115650, 9.6887207, 8.2223282, 9.6604738, -1.0437448, 1.0526855
6: -4.7691312, -2.3143466, -4.7883987, -2.3185630, -1.6572723, 1.5295680
7: -15.7503271, -12.9480457, -15.7886810, -12.9930592, -1.8582425, 2.2050381
8: -0.8169909, 0.9446225, -0.7905314, 0.9714684, -1.2399139, 1.0787098
9: -6.6713853, -5.0047336, -6.6747465, -4.9597888, -1.5555830, 1.2085605

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6689649
time: 4.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6596229
time: 4.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.3020611, -10.5113811, -13.1615391, -10.5563307, -1.6449695, 1.5758913
1: -11.4193535, -8.4564342, -11.3216839, -8.4316435, -2.0331111, 1.8683579
2: -10.7133245, -8.5717316, -10.7237625, -8.5686007, -1.7004328, 1.6745262
3: -4.5645933, -2.3432310, -4.3982882, -2.3434389, -1.5553637, 1.5376210
4: -15.1517258, -12.4743414, -15.1122036, -12.5866070, -1.6747875, 1.5971518
5: 8.2365532, 9.6461134, 8.2518520, 9.6990671, -1.0577247, 0.9806726
6: -4.7766171, -2.3436375, -4.7113757, -2.3275948, -1.5489669, 1.5366077
7: -15.7843122, -12.9983559, -15.7493277, -12.9501801, -2.1986036, 1.8363237
8: -0.7559652, 0.9712560, -0.7736735, 0.9030194, -1.0471044, 1.1843629
9: -6.6707249, -4.9770184, -6.6735115, -5.0673761, -1.1040621, 1.5672264

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6407374
time: 3.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6426902
time: 3.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.3020611, -10.5113811, -13.3071823, -10.5114536, -1.5450342, 1.5290985
1: -11.4193535, -8.4564342, -11.4477100, -8.4399529, -1.8832250, 1.8730514
2: -10.7133245, -8.5717316, -10.7732630, -8.5800972, -1.6155341, 1.6679332
3: -4.5645933, -2.3432310, -4.5499749, -2.3376729, -1.3501594, 1.3436992
4: -15.1517258, -12.4743414, -15.1255608, -12.5076370, -1.5048149, 1.5184679
5: 8.2365532, 9.6461134, 8.2469168, 9.6606894, -0.9722962, 0.9505900
6: -4.7766171, -2.3436375, -4.7614303, -2.3335683, -1.4634266, 1.4350536
7: -15.7843122, -12.9983559, -15.7939177, -12.9961119, -1.8292570, 1.8391683
8: -0.7559652, 0.9712560, -0.7472873, 0.9561176, -1.0124433, 1.0188833
9: -6.6707249, -4.9770184, -6.6714001, -5.0126677, -1.1300542, 1.1465583

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6407374
time: 3.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6426903
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3320122, -10.4996128, -13.1625299, -10.5510769, -1.7089291, 1.5872214
1: -11.4292049, -8.4717159, -11.3229933, -8.4421015, -2.1256375, 1.8659389
2: -10.7522335, -8.5140104, -10.7377396, -8.5672417, -1.7178335, 1.7880955
3: -4.5708861, -2.3397861, -4.3987608, -2.3430641, -1.5610290, 1.5404940
4: -15.1485224, -12.4820595, -15.1122456, -12.5902300, -1.6682587, 1.6003084
5: 8.2239780, 9.6503048, 8.2516966, 9.6995344, -1.0693272, 0.9919292
6: -4.8073292, -2.3203678, -4.7117066, -2.3215444, -1.6082466, 1.5466869
7: -15.7946587, -12.9966345, -15.7503586, -12.9508038, -2.2013397, 1.8409150
8: -0.7747347, 0.9977977, -0.7791660, 0.9030747, -1.0695159, 1.2092991
9: -6.6690583, -4.9738240, -6.6720467, -5.0668530, -1.1079631, 1.5794554

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6459143
time: 4.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6478674
time: 3.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3320122, -10.4996128, -13.3082132, -10.5061989, -1.6091356, 1.5381877
1: -11.4292049, -8.4717159, -11.4490948, -8.4504070, -1.9755235, 1.8706193
2: -10.7522335, -8.5140104, -10.7872429, -8.5788212, -1.6330187, 1.7813289
3: -4.5708861, -2.3397861, -4.5504627, -2.3372984, -1.3557198, 1.3471293
4: -15.1485224, -12.4820595, -15.1256056, -12.5112572, -1.5094018, 1.5215468
5: 8.2239780, 9.6503048, 8.2467575, 9.6611557, -0.9894140, 0.9618441
6: -4.8073292, -2.3203678, -4.7617340, -2.3275123, -1.5226297, 1.4472005
7: -15.7946587, -12.9966345, -15.7951097, -12.9967470, -1.8395662, 1.8436465
8: -0.7747347, 0.9977977, -0.7527628, 0.9561768, -1.0350249, 1.0489918
9: -6.6690583, -4.9738240, -6.6699343, -5.0121303, -1.1339469, 1.1729679

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6459143
time: 3.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6478673
time: 3.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3020706, -10.5055943, -13.1595364, -10.5442724, -1.6362159, 1.5859809
1: -11.4193573, -8.4571171, -11.2972002, -8.4385118, -2.0424161, 1.8503528
2: -10.7133274, -8.5628185, -10.6993294, -8.5630732, -1.6796811, 1.7162054
3: -4.5753388, -2.3432283, -4.4175091, -2.3477132, -1.5805073, 1.5291672
4: -15.1517277, -12.4306383, -15.1384869, -12.5098171, -1.6732697, 1.6469190
5: 8.2190933, 9.6461124, 8.2275000, 9.6984005, -1.0774379, 0.9836051
6: -4.7891903, -2.3436341, -4.7380171, -2.3186455, -1.5734615, 1.5295124
7: -15.7843113, -12.9957619, -15.7429028, -12.9463787, -2.2147617, 1.8247333
8: -0.7716520, 0.9712572, -0.8114500, 0.9183147, -1.0802853, 1.1760325
9: -6.6707234, -4.9532433, -6.6784458, -5.0150347, -1.0850973, 1.5750656

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6571215
time: 5.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6477840
time: 4.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3020706, -10.5055943, -13.3050594, -10.4993925, -1.5362811, 1.5508835
1: -11.4193573, -8.4571171, -11.4232244, -8.4468231, -1.8925295, 1.8550308
2: -10.7133274, -8.5628185, -10.7488317, -8.5745049, -1.5949159, 1.7096133
3: -4.5753388, -2.3432283, -4.5692968, -2.3419483, -1.3752332, 1.3417916
4: -15.1517277, -12.4306383, -15.1518402, -12.4307404, -1.5144355, 1.5681746
5: 8.2190933, 9.6461124, 8.2224941, 9.6599913, -0.9975152, 0.9535215
6: -4.7891903, -2.3436341, -4.7880993, -2.3246164, -1.4878912, 1.4318779
7: -15.7843113, -12.9957619, -15.7874365, -12.9924278, -1.8509064, 1.8275516
8: -0.7716520, 0.9712572, -0.7850575, 0.9714088, -1.0456064, 1.0137035
9: -6.6707234, -4.9532433, -6.6762142, -4.9603271, -1.1110907, 1.2081645

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6571214
time: 5.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6477839
time: 4.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3320208, -10.4938269, -13.1605358, -10.5390167, -1.7002153, 1.5972908
1: -11.4292088, -8.4723969, -11.2985115, -8.4489708, -2.1349068, 1.8479438
2: -10.7522354, -8.5051794, -10.7133093, -8.5617695, -1.6970885, 1.8297708
3: -4.5816317, -2.3397834, -4.4179573, -2.3473399, -1.5861750, 1.5320301
4: -15.1485252, -12.4383535, -15.1385279, -12.5134401, -1.6667295, 1.6500821
5: 8.2065287, 9.6503048, 8.2273388, 9.6988859, -1.0890497, 0.9948120
6: -4.8199048, -2.3203635, -4.7383471, -2.3125937, -1.6327543, 1.5395792
7: -15.7946606, -12.9940453, -15.7439833, -12.9470024, -2.2174034, 1.8293328
8: -0.7904239, 0.9978006, -0.8169415, 0.9183688, -1.1026967, 1.2009673
9: -6.6690540, -4.9500480, -6.6769814, -5.0145102, -1.0889983, 1.5872846

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6623016
time: 4.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6529610
time: 4.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3320208, -10.4938269, -13.3060932, -10.4941368, -1.6004200, 1.5599961
1: -11.4292088, -8.4723969, -11.4246092, -8.4572773, -1.9847922, 1.8526077
2: -10.7522354, -8.5051794, -10.7628136, -8.5732822, -1.6124058, 1.8230021
3: -4.5816317, -2.3397834, -4.5697613, -2.3415761, -1.3807936, 1.3452184
4: -15.1485252, -12.4383535, -15.1518841, -12.4343634, -1.5190387, 1.5712597
5: 8.2065287, 9.6503048, 8.2223282, 9.6604738, -1.0147116, 0.9647243
6: -4.8199048, -2.3203635, -4.7883987, -2.3185630, -1.5471089, 1.4441748
7: -15.7946606, -12.9940453, -15.7886810, -12.9930592, -1.8611770, 1.8320379
8: -0.7904239, 0.9978006, -0.7905314, 0.9714684, -1.0681887, 1.0438234
9: -6.6690540, -4.9500480, -6.6747465, -4.9597888, -1.1149817, 1.2345803

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6623012
time: 5.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6529611
time: 3.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.15 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6473993
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6493521
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6473992
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6493515
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6525281
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6545074
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6525281
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6545074
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6637877
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6544458
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6637876
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6544458
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6689646
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6596230
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6689649
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6596229
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6407374
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6426902
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6407374
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6512548, upper bound: 0.6426903
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6459143
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6478674
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6459143
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6478673
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6571215
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6477840
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6571214
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6477839
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6623016
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6529610
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6623012
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.15
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6529611

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1509151, -10.5577116, -13.1606236, -10.5563326, -1.5537477, 1.5425594
1: -11.2137823, -8.4934368, -11.2950773, -8.4316435, -1.7674961, 1.7061243
2: -10.6471481, -8.5696726, -10.7184429, -8.5696888, -1.5928895, 1.6538069
3: -4.4175172, -2.3639085, -4.3979673, -2.3494520, -1.5284824, 1.5199552
4: -15.1261492, -12.5309010, -15.1072817, -12.5868454, -1.6637139, 1.6787658
5: 8.2421799, 9.6747904, 8.2522888, 9.6929188, -0.9757628, 0.9616337
6: -4.7253470, -2.3384385, -4.7110443, -2.3276808, -1.5664468, 1.5385265
7: -15.7133188, -12.9726601, -15.7370777, -12.9505301, -2.0784798, 2.0838947
8: -0.7406323, 0.8842683, -0.7600408, 0.9029198, -1.0079269, 1.0001278
9: -6.6702366, -5.0353165, -6.6733022, -5.0679350, -1.4652805, 1.5028195

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6587976
time: 3.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6587975
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1559381, -10.5562601, -13.1613102, -10.5563335, -1.5583835, 1.5466521
1: -11.2646894, -8.4481239, -11.3109465, -8.4316435, -1.6871729, 1.8431931
2: -10.6497011, -8.5607204, -10.7202597, -8.5687141, -1.6070805, 1.6516519
3: -4.4123926, -2.3731034, -4.3981276, -2.3497500, -1.5377350, 1.5090034
4: -15.1083059, -12.5537148, -15.1045818, -12.5867338, -1.6396484, 1.6944747
5: 8.2420378, 9.6690598, 8.2520409, 9.6969004, -1.0004721, 0.9428428
6: -4.7259097, -2.3377416, -4.7112174, -2.3276215, -1.5678940, 1.5392723
7: -15.7240696, -12.9532633, -15.7468319, -12.9504290, -2.0720854, 2.1151538
8: -0.7754569, 0.9180484, -0.7714250, 0.9029791, -0.9909439, 1.0484006
9: -6.6723557, -5.0321651, -6.6733408, -5.0675435, -1.4684048, 1.5049930

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6607890
time: 4.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6607890
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1509151, -10.5577116, -13.3063812, -10.5114546, -1.5811152, 1.6225274
1: -11.2137823, -8.4934368, -11.4211216, -8.4399529, -1.7907891, 1.8845339
2: -10.6471481, -8.5696726, -10.7679386, -8.5811996, -1.5969882, 1.7359359
3: -4.4175172, -2.3639085, -4.5498400, -2.3436921, -1.5262413, 1.5293732
4: -15.1261492, -12.5309010, -15.1206474, -12.5077362, -1.5660801, 1.6712873
5: 8.2421799, 9.6747904, 8.2473316, 9.6545916, -0.9816933, 1.0180554
6: -4.7253470, -2.3384385, -4.7611971, -2.3336582, -1.5635633, 1.5182648
7: -15.7133188, -12.9726601, -15.7817259, -12.9962006, -1.7910647, 2.1782637
8: -0.7406323, 0.8842683, -0.7336490, 0.9560232, -1.1416655, 0.9992342
9: -6.6702366, -5.0353165, -6.6713924, -5.0132322, -1.5230241, 1.1166351

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6473992
time: 3.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6473993
time: 4.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1559381, -10.5562601, -13.3070393, -10.5114536, -1.5857506, 1.6273198
1: -11.2646894, -8.4481239, -11.4369659, -8.4399529, -1.7104664, 2.0215697
2: -10.6497011, -8.5607204, -10.7697544, -8.5802078, -1.6117687, 1.7337799
3: -4.4123926, -2.3731034, -4.5499640, -2.3439937, -1.5355577, 1.5130522
4: -15.1083059, -12.5537148, -15.1179466, -12.5076475, -1.5438962, 1.6869905
5: 8.2420378, 9.6690598, 8.2470951, 9.6585627, -1.0017567, 0.9992650
6: -4.7259097, -2.3377416, -4.7613554, -2.3335958, -1.5650120, 1.5184460
7: -15.7240696, -12.9532633, -15.7914181, -12.9961433, -1.7747955, 2.2094259
8: -0.7754569, 0.9180484, -0.7450824, 0.9560792, -1.1247168, 1.0515916
9: -6.6723557, -5.0321651, -6.6713996, -5.0128379, -1.5261250, 1.1194043

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6493520
time: 4.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6493521
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1807270, -10.5459461, -13.1616039, -10.5510769, -1.6225624, 1.5538845
1: -11.2236605, -8.5087233, -11.2963820, -8.4421015, -1.8597803, 1.7037091
2: -10.6860886, -8.5120144, -10.7324209, -8.5683546, -1.6102538, 1.7669704
3: -4.4238062, -2.3604650, -4.3984351, -2.3490765, -1.5354667, 1.5229270
4: -15.1229277, -12.5386000, -15.1073265, -12.5904713, -1.6569014, 1.6823618
5: 8.2297153, 9.6790104, 8.2521343, 9.6933880, -0.9864693, 0.9733019
6: -4.7555709, -2.3151402, -4.7113748, -2.3216300, -1.6294761, 1.5486023
7: -15.7235003, -12.9710178, -15.7381115, -12.9511557, -2.0807877, 2.0836382
8: -0.7595429, 0.9107118, -0.7655389, 0.9029751, -1.0316932, 1.0248668
9: -6.6686320, -5.0321064, -6.6718369, -5.0674090, -1.4659152, 1.5149922

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6616710
time: 3.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6630128
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1857052, -10.5444927, -13.1622982, -10.5510798, -1.6272235, 1.5579870
1: -11.2744026, -8.4634132, -11.3122597, -8.4421015, -1.7793770, 1.8407753
2: -10.6886358, -8.5031853, -10.7342434, -8.5673590, -1.6243482, 1.7650526
3: -4.4186516, -2.3696573, -4.3985987, -2.3493757, -1.5447307, 1.5119672
4: -15.1050854, -12.5614281, -15.1046247, -12.5903549, -1.6328034, 1.6980655
5: 8.2295103, 9.6732788, 8.2518864, 9.6973686, -1.0110818, 0.9544688
6: -4.7560315, -2.3144412, -4.7115469, -2.3215699, -1.6308627, 1.5493636
7: -15.7342033, -12.9515762, -15.7478647, -12.9510517, -2.0745573, 2.1149278
8: -0.7943101, 0.9444897, -0.7769165, 0.9030342, -1.0147052, 1.0731423
9: -6.6707573, -5.0289803, -6.6718750, -5.0670195, -1.4690542, 1.5171247

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6636542
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6650788
time: 4.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1807270, -10.5459461, -13.3074007, -10.5061989, -1.6499305, 1.6316926
1: -11.2236605, -8.5087233, -11.4225044, -8.4504070, -1.8830719, 1.8823118
2: -10.6860886, -8.5120144, -10.7819185, -8.5799465, -1.6145492, 1.8490958
3: -4.4238062, -2.3604650, -4.5503235, -2.3433189, -1.5332489, 1.5328600
4: -15.1229277, -12.5386000, -15.1206875, -12.5113592, -1.5703821, 1.6748559
5: 8.2297153, 9.6790104, 8.2471733, 9.6550579, -0.9979699, 1.0297358
6: -4.7555709, -2.3151402, -4.7614932, -2.3276012, -1.6265945, 1.5303924
7: -15.7235003, -12.9710178, -15.7829199, -12.9968328, -1.8013902, 2.1782761
8: -0.7595429, 0.9107118, -0.7391317, 0.9560823, -1.1654305, 1.0292444
9: -6.6686320, -5.0321064, -6.6699252, -5.0126896, -1.5237117, 1.1430640

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6503584
time: 4.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6525283
time: 4.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1857052, -10.5444927, -13.3080673, -10.5062008, -1.6545911, 1.6364756
1: -11.2744026, -8.4634132, -11.4383554, -8.4504070, -1.8026695, 2.0193481
2: -10.6886358, -8.5031853, -10.7837420, -8.5789337, -1.6292181, 1.8471782
3: -4.4186516, -2.3696573, -4.5504503, -2.3436203, -1.5425763, 1.5165427
4: -15.1050854, -12.5614281, -15.1179924, -12.5112648, -1.5481472, 1.6905527
5: 8.2295103, 9.6732788, 8.2469349, 9.6590309, -1.0178530, 1.0109029
6: -4.7560315, -2.3144412, -4.7616587, -2.3275394, -1.6279821, 1.5305541
7: -15.7342033, -12.9515762, -15.7926159, -12.9967785, -1.7851429, 2.2094669
8: -0.7943101, 0.9444897, -0.7505603, 0.9561391, -1.1484752, 1.0816140
9: -6.6707573, -5.0289803, -6.6699319, -5.0122991, -1.5268269, 1.1457958

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6523114
time: 3.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6545076
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.1595364, -10.5442724, -1.5745964, 1.5400991
1: -11.3179722, -8.4517279, -11.2972002, -8.4385118, -1.8637991, 1.8253074
2: -10.6882601, -8.5720196, -10.6993294, -8.5630732, -1.6476388, 1.6625557
3: -4.3971014, -2.3447456, -4.4175091, -2.3477132, -1.5406446, 1.5563710
4: -15.1120892, -12.5913963, -15.1384869, -12.5098171, -1.7316294, 1.6804795
5: 8.2523460, 9.6852093, 8.2275000, 9.6984005, -0.9871137, 1.0035219
6: -4.7101393, -2.3466036, -4.7380171, -2.3186455, -1.5647578, 1.5576053
7: -15.7467308, -12.9583788, -15.7429028, -12.9463787, -2.1173935, 2.1002197
8: -0.7583771, 0.9028707, -0.8114500, 0.9183147, -1.0386972, 1.0764968
9: -6.6679935, -5.0688710, -6.6784458, -5.0150347, -1.5139952, 1.4769907

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6615289, upper bound: 0.6663797
time: 4.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6612244, upper bound: 0.6663797
time: 4.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.1595364, -10.5442724, -1.5615811, 1.5401959
1: -11.2934866, -8.4585981, -11.2972002, -8.4385118, -1.8645201, 1.8345721
2: -10.6638260, -8.5663605, -10.6993294, -8.5630732, -1.5974858, 1.6418040
3: -4.4163847, -2.3490036, -4.4175091, -2.3477132, -1.5322227, 1.5314703
4: -15.1383724, -12.5146065, -15.1384869, -12.5098171, -1.6806955, 1.6789863
5: 8.2280121, 9.6844997, 8.2275000, 9.6984005, -0.9938145, 0.9846730
6: -4.7367697, -2.3376508, -4.7380171, -2.3186455, -1.5576963, 1.5323777
7: -15.7401762, -12.9545841, -15.7429028, -12.9463787, -2.1211300, 2.1162224
8: -0.7961533, 0.9181619, -0.8114500, 0.9183147, -1.0303690, 1.0422649
9: -6.6729279, -5.0165362, -6.6784458, -5.0150347, -1.4971189, 1.5014501

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6615289, upper bound: 0.6593350
time: 4.14 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6612244, upper bound: 0.6593351
time: 3.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.3050594, -10.4993925, -1.6019630, 1.6276474
1: -11.3179722, -8.4517279, -11.4232244, -8.4468231, -1.8870935, 2.0036511
2: -10.6882601, -8.5720196, -10.7488317, -8.5745049, -1.6509895, 1.7446842
3: -4.3971014, -2.3447456, -4.5692968, -2.3419483, -1.5384154, 1.5638347
4: -15.1120892, -12.5913963, -15.1518402, -12.4307404, -1.6299891, 1.6729898
5: 8.2523460, 9.6852093, 8.2224941, 9.6599913, -0.9913127, 1.0599722
6: -4.7101393, -2.3466036, -4.7880993, -2.3246164, -1.5618792, 1.5375426
7: -15.7467308, -12.9583788, -15.7874365, -12.9924278, -1.8404493, 2.1944194
8: -0.7583771, 0.9028707, -0.7850575, 0.9714088, -1.1724343, 1.0770414
9: -6.6679935, -5.0688710, -6.6762142, -4.9603271, -1.5717211, 1.1121597

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6435404, upper bound: 0.6549591
time: 4.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6431540, upper bound: 0.6549592
time: 3.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1566315, -10.5564013, -13.3050594, -10.4993925, -1.5889473, 1.6188478
1: -11.2934866, -8.4585981, -11.4232244, -8.4468231, -1.8878140, 2.0129342
2: -10.6638260, -8.5663605, -10.7488317, -8.5745049, -1.6017389, 1.7239337
3: -4.4163847, -2.3490036, -4.5692968, -2.3419483, -1.5300384, 1.5471213
4: -15.1383724, -12.5146065, -15.1518402, -12.4307404, -1.5932341, 1.6715176
5: 8.2280121, 9.6844997, 8.2224941, 9.6599913, -0.9941571, 1.0410891
6: -4.7367697, -2.3376508, -4.7880993, -2.3246164, -1.5548153, 1.5172925
7: -15.7401762, -12.9545841, -15.7874365, -12.9924278, -1.8480077, 2.2104883
8: -0.7961533, 0.9181619, -0.7850575, 0.9714088, -1.1641068, 1.0486567
9: -6.6729279, -5.0165362, -6.6762142, -4.9603271, -1.5548420, 1.0932093

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6435404, upper bound: 0.6478791
time: 4.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6431540, upper bound: 0.6478772
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.1605358, -10.5390167, -1.6433144, 1.5514095
1: -11.3276834, -8.4670172, -11.2985115, -8.4489708, -1.9556875, 1.8228981
2: -10.7271967, -8.5144558, -10.7133093, -8.5617695, -1.6649866, 1.7759454
3: -4.4033794, -2.3413148, -4.4179573, -2.3473399, -1.5476232, 1.5593328
4: -15.1088705, -12.5990896, -15.1385279, -12.5134401, -1.7247930, 1.6840758
5: 8.2397890, 9.6893950, 8.2273388, 9.6988859, -0.9977990, 1.0151045
6: -4.7403202, -2.3233070, -4.7383471, -2.3125937, -1.6277618, 1.5676866
7: -15.7568035, -12.9566870, -15.7439833, -12.9470024, -2.1197186, 2.1000700
8: -0.7772584, 0.9293244, -0.8169415, 0.9183688, -1.0623589, 1.1012552
9: -6.6664100, -5.0656857, -6.6769814, -5.0145102, -1.5146580, 1.4891067

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6629008, upper bound: 0.6706706
time: 4.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6626789, upper bound: 0.6706708
time: 4.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.1605358, -10.5390167, -1.6302977, 1.5515258
1: -11.3031998, -8.4738846, -11.2985115, -8.4489708, -1.9564075, 1.8321536
2: -10.7027636, -8.5088758, -10.7133093, -8.5617695, -1.6148832, 1.7551923
3: -4.4226265, -2.3455589, -4.4179573, -2.3473399, -1.5392036, 1.5344441
4: -15.1351519, -12.5223141, -15.1385279, -12.5134401, -1.6738639, 1.6825857
5: 8.2154884, 9.6887102, 8.2273388, 9.6988859, -1.0044825, 0.9962519
6: -4.7669148, -2.3143487, -4.7383471, -2.3125937, -1.6207018, 1.5424616
7: -15.7503185, -12.9529018, -15.7439833, -12.9470024, -2.1234512, 2.1160097
8: -0.8150344, 0.9446130, -0.8169415, 0.9183688, -1.0540326, 1.0670229
9: -6.6713448, -5.0133581, -6.6769814, -5.0145102, -1.4977818, 1.5135660

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6629008, upper bound: 0.6626809
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6626789, upper bound: 0.6637491
time: 4.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.3060932, -10.4941368, -1.6706815, 1.6368377
1: -11.3276834, -8.4670172, -11.4246092, -8.4572773, -1.9789801, 2.0014353
2: -10.7271967, -8.5144558, -10.7628136, -8.5732822, -1.6685271, 1.8580704
3: -4.4033794, -2.3413148, -4.5697613, -2.3415761, -1.5454178, 1.5673029
4: -15.1088705, -12.5990896, -15.1518841, -12.4343634, -1.6342864, 1.6765571
5: 8.2397890, 9.6893950, 8.2223282, 9.6604738, -1.0075405, 1.0715663
6: -4.7403202, -2.3233070, -4.7883987, -2.3185630, -1.6248841, 1.5498242
7: -15.7568035, -12.9566870, -15.7886810, -12.9930592, -1.8506784, 2.1945381
8: -0.7772584, 0.9293244, -0.7905314, 0.9714684, -1.1960931, 1.1071005
9: -6.6664100, -5.0656857, -6.6747465, -4.9597888, -1.5724354, 1.1385612

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6457583, upper bound: 0.6601005
time: 4.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6453719, upper bound: 0.6600987
time: 6.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1863823, -10.5446339, -13.3060932, -10.4941368, -1.6576643, 1.6280141
1: -11.3031998, -8.4738846, -11.4246092, -8.4572773, -1.9797006, 2.0107100
2: -10.7027636, -8.5088758, -10.7628136, -8.5732822, -1.6193323, 1.8373175
3: -4.4226265, -2.3455589, -4.5697613, -2.3415761, -1.5370426, 1.5506077
4: -15.1351519, -12.5223141, -15.1518841, -12.4343634, -1.5975299, 1.6750884
5: 8.2154884, 9.6887102, 8.2223282, 9.6604738, -1.0103459, 1.0526798
6: -4.7669148, -2.3143487, -4.7883987, -2.3185630, -1.6178207, 1.5295670
7: -15.7503185, -12.9529018, -15.7886810, -12.9930592, -1.8582387, 2.2105446
8: -0.8150344, 0.9446130, -0.7905314, 0.9714684, -1.1877675, 1.0786986
9: -6.6713448, -5.0133581, -6.6747465, -4.9597888, -1.5555563, 1.1196103

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6457583, upper bound: 0.6530414
time: 4.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6453719, upper bound: 0.6530416
time: 3.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.2966709, -10.5128307, -13.1606236, -10.5563326, -1.6362195, 1.5699294
1: -11.3394489, -8.5017471, -11.2950773, -8.4316435, -1.9448657, 1.7294190
2: -10.6966333, -8.5806484, -10.7184429, -8.5696888, -1.6750286, 1.6574512
3: -4.5698223, -2.3581710, -4.3979673, -2.3494520, -1.5343585, 1.5176129
4: -15.1394901, -12.4515133, -15.1072817, -12.5868454, -1.6562572, 1.5768530
5: 8.2372494, 9.6365433, 8.2522888, 9.6929188, -1.0322778, 0.9662651
6: -4.7758479, -2.3444228, -4.7110443, -2.3276808, -1.5467691, 1.5356419
7: -15.7571383, -13.0177889, -15.7370777, -12.9505301, -2.1715412, 1.7867022
8: -0.7143154, 0.9373667, -0.7600408, 0.9029198, -1.0111213, 1.1338894
9: -6.6686625, -4.9806900, -6.6733022, -5.0679350, -1.1001923, 1.5603123

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6407357
time: 4.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6407374
time: 4.14 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.3016529, -10.5113831, -13.1613102, -10.5563335, -1.6404293, 1.5740182
1: -11.3905420, -8.4564342, -11.3109465, -8.4316435, -1.8648801, 1.8664870
2: -10.6991825, -8.5719957, -10.7202597, -8.5687141, -1.6892242, 1.6550772
3: -4.5645704, -2.3673952, -4.3981276, -2.3497500, -1.5489168, 1.5065405
4: -15.1216373, -12.4743690, -15.1045818, -12.5867338, -1.6321402, 1.5929742
5: 8.2370796, 9.6308079, 8.2520409, 9.6969004, -1.0569434, 0.9487495
6: -4.7763915, -2.3437297, -4.7112174, -2.3276215, -1.5483594, 1.5363891
7: -15.7682114, -12.9984484, -15.7468319, -12.9504290, -2.1655817, 1.8352957
8: -0.7491698, 0.9711442, -0.7714250, 0.9029791, -0.9914722, 1.1821461
9: -6.6707191, -4.9775019, -6.6733408, -5.0675435, -1.1026807, 1.5625734

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6426903
time: 3.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6426902
time: 3.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.2966709, -10.5128307, -13.3063812, -10.5114546, -1.5363536, 1.5224028
1: -11.3394489, -8.5017471, -11.4211216, -8.4399529, -1.7953501, 1.7340772
2: -10.6966333, -8.5806484, -10.7679386, -8.5811996, -1.5900664, 1.6510482
3: -4.5698223, -2.3581710, -4.5498400, -2.3436921, -1.3292468, 1.3239202
4: -15.1394901, -12.4515133, -15.1206474, -12.5077362, -1.4872131, 1.4982798
5: 8.2372494, 9.6365433, 8.2473316, 9.6545916, -0.9517568, 0.9361274
6: -4.7758479, -2.3444228, -4.7611971, -2.3336582, -1.4612055, 1.4328218
7: -15.7571383, -13.0177889, -15.7817259, -12.9962006, -1.7939024, 1.7895443
8: -0.7143154, 0.9373667, -0.7336490, 0.9560232, -0.9764464, 0.9642732
9: -6.6686625, -4.9806900, -6.6713924, -5.0132322, -1.1261716, 1.1426871

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6407374
time: 4.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6407373
time: 4.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.3016529, -10.5113831, -13.3070393, -10.5114536, -1.5405111, 1.5272024
1: -11.3905420, -8.4564342, -11.4369659, -8.4399529, -1.7149699, 1.8711886
2: -10.6991825, -8.5719957, -10.7697544, -8.5802078, -1.6048949, 1.6484818
3: -4.5645704, -2.3673952, -4.5499640, -2.3439937, -1.3437181, 1.3075740
4: -15.1216373, -12.4743690, -15.1179466, -12.5076475, -1.4650183, 1.5142913
5: 8.2370796, 9.6308079, 8.2470951, 9.6585627, -0.9717010, 0.9186926
6: -4.7763915, -2.3437297, -4.7613554, -2.3335958, -1.4628196, 1.4329988
7: -15.7682114, -12.9984484, -15.7914181, -12.9961433, -1.7775941, 1.8381317
8: -0.7491698, 0.9711442, -0.7450824, 0.9560792, -0.9568279, 1.0166090
9: -6.6707191, -4.9775019, -6.6713996, -5.0128379, -1.1286745, 1.1454155

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6426903
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6426902
time: 3.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.3266678, -10.5010643, -13.1616039, -10.5510769, -1.7002907, 1.5812559
1: -11.3494844, -8.5170259, -11.2963820, -8.4421015, -2.0377960, 1.7270033
2: -10.7355461, -8.5228453, -10.7324209, -8.5683546, -1.6924019, 1.7708781
3: -4.5761480, -2.3547239, -4.3984351, -2.3490765, -1.5400505, 1.5204766
4: -15.1362886, -12.4592171, -15.1073265, -12.5904713, -1.6497383, 1.5800200
5: 8.2247381, 9.6407452, 8.2521343, 9.6933880, -1.0439279, 0.9775951
6: -4.8066330, -2.3211560, -4.7113748, -2.3216300, -1.6060989, 1.5456991
7: -15.7675304, -13.0161114, -15.7381115, -12.9511557, -2.1742105, 1.7912998
8: -0.7331181, 0.9639027, -0.7655389, 0.9029751, -1.0337560, 1.1588087
9: -6.6669927, -4.9774599, -6.6718369, -5.0674090, -1.1040919, 1.5726075

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6436967
time: 3.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6459144
time: 3.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.3315973, -10.4996147, -13.1622982, -10.5510798, -1.7044396, 1.5853536
1: -11.4003935, -8.4717159, -11.3122597, -8.4421015, -1.9576969, 1.8640690
2: -10.7380905, -8.5143051, -10.7342434, -8.5673590, -1.7065001, 1.7687349
3: -4.5708604, -2.3639469, -4.3985987, -2.3493757, -1.5546045, 1.5093961
4: -15.1184368, -12.4820833, -15.1046247, -12.5903549, -1.6255865, 1.5961401
5: 8.2245083, 9.6350098, 8.2518864, 9.6973686, -1.0684751, 0.9600519
6: -4.8070698, -2.3204637, -4.7115469, -2.3215699, -1.6076136, 1.5464621
7: -15.7785530, -12.9967289, -15.7478647, -12.9510517, -2.1683798, 1.8398840
8: -0.7679167, 0.9976778, -0.7769165, 0.9030342, -1.0140731, 1.2070661
9: -6.6690516, -4.9743009, -6.6718750, -5.0670195, -1.1065850, 1.5748181

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6456496
time: 3.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6478674
time: 3.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.06 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6587976
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6587975
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6607890
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6664487, upper bound: 0.6607890
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6473992
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6473993
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6493520
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6493521
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6616710
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6630128
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6636542
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6663777, upper bound: 0.6650788
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6503584
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6525283
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6523114
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6482955, upper bound: 0.6545076
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6615289, upper bound: 0.6663797
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6612244, upper bound: 0.6663797
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6615289, upper bound: 0.6593350
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6612244, upper bound: 0.6593351
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6435404, upper bound: 0.6549591
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6431540, upper bound: 0.6549592
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6435404, upper bound: 0.6478791
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6431540, upper bound: 0.6478772
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6629008, upper bound: 0.6706706
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6626789, upper bound: 0.6706708
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6629008, upper bound: 0.6626809
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6626789, upper bound: 0.6637491
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6457583, upper bound: 0.6601005
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6453719, upper bound: 0.6600987
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6457583, upper bound: 0.6530414
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6453719, upper bound: 0.6530416
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6407357
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6407374
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6426903
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6550285, upper bound: 0.6426902
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6407374
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6407373
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6426903
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6483667, upper bound: 0.6426902
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6436967
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6459144
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6456496
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.06
Output dim: 5, lower bound: -0.6549573, upper bound: 0.6478674
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6459143
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6534727, upper bound: 0.6478673
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6571215
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6477840
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6571214
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6507412, upper bound: 0.6477839
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6623016
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6529610
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6623012
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.06
Output dim: 5, lower bound: -0.6529591, upper bound: 0.6529611
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.0726536512374878
rel_dist={5: [-0.7059536409330516, 0.7059557061852892]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2412.37 seconds
