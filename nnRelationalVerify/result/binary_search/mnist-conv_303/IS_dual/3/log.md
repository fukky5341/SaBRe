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
execution time: IAR + LP analysis = 14.91 + 32.12 = 47.03 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.97 seconds, max iter: 100)

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
Binary search time: 152.41 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual) starts
Time budget: 3400.56 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9828011
time: 3.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687710
time: 4.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.14
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9828011
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.14
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

Time for backsubstitution: 5.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687699
time: 4.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687705
time: 4.19 seconds

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

Time for backsubstitution: 5.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687718
time: 3.61 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687719
time: 3.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 12.49
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687699
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.49
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687705
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.49
Output dim: 5, lower bound: -0.9687700, upper bound: 0.9687718
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.49
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

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9718186
time: 4.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9718645
time: 4.88 seconds

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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9534195
time: 3.82 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578351, upper bound: 0.9718658
time: 3.76 seconds

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

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577891
time: 5.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369
time: 3.64 seconds

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

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577911
time: 3.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369
time: 4.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9718186
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9718645
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9534195
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9578351, upper bound: 0.9718658
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577891
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9578349, upper bound: 0.9578369
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.57
Output dim: 5, lower bound: -0.9393885, upper bound: 0.9577911
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.57
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

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674947
time: 3.52 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9496548, upper bound: 0.9680572
time: 4.00 seconds

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

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624619
time: 3.71 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624618
time: 3.53 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.1613808, -10.5297728, -13.3090773, -10.5022688, -2.0899982, 2.1713259
1: -11.2995062, -8.4145632, -11.4500942, -8.4280024, -2.4462223, 2.6225243
2: -10.7255230, -8.5475016, -10.7994423, -8.5782375, -2.0867276, 2.1867943
3: -4.4236903, -2.3468392, -4.5506849, -2.3368161, -1.9228182, 1.9386008
4: -15.1385679, -12.5178137, -15.1256447, -12.5048466, -2.1441574, 2.2587261
5: 8.2293205, 9.7064915, 8.2465706, 9.6687326, -1.2937901, 1.3230305
6: -4.7367353, -2.3063402, -4.7621164, -2.3212516, -2.0247259, 2.0211263
7: -15.7444878, -12.9369030, -15.7956581, -12.9913330, -2.3403573, 2.7228875
8: -0.8168800, 0.9184299, -0.7562866, 0.9562299, -1.5037651, 1.3522345
9: -6.6816845, -5.0127063, -6.6745906, -5.0117588, -1.6699257, 1.4459434

Time for backsubstitution: 5.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9326082
time: 3.52 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9393824
time: 3.62 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.3069401, -10.4902096, -2.0928640, 2.1869287
1: -11.2995148, -8.4136467, -11.4256086, -8.4348717, -2.4479284, 2.6064515
2: -10.7255239, -8.5416927, -10.7750120, -8.5727234, -2.0636506, 2.2157500
3: -4.4290442, -2.3468359, -4.5699735, -2.3410995, -1.9455805, 1.9444649
4: -15.1385698, -12.5005589, -15.1519194, -12.4279480, -2.1796455, 2.2932301
5: 8.2217073, 9.7064924, 8.2221375, 9.6680603, -1.3092346, 1.3408229
6: -4.7417126, -2.3063393, -4.7887893, -2.3123035, -2.0441852, 2.0249281
7: -15.7444887, -12.9351664, -15.7892513, -12.9876451, -2.3697252, 2.7151060
8: -0.8229923, 0.9184313, -0.7940555, 0.9715207, -1.5281458, 1.3584472
9: -6.6816912, -5.0027695, -6.6794038, -4.9594150, -1.7222762, 1.4862170

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9381566, upper bound: 0.9718202
time: 3.47 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9381566, upper bound: 0.9718661
time: 3.50 seconds

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

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
time: 3.90 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9393804, upper bound: 0.9437539
time: 3.75 seconds

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

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9381584
time: 3.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9578369
time: 3.63 seconds

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

Time for backsubstitution: 5.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
time: 3.65 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9253513, upper bound: 0.9437539
time: 3.66 seconds

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

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9381585
time: 3.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9578370
time: 3.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 12.87 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674947
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9496548, upper bound: 0.9680572
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624619
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9820927, upper bound: 0.9624618
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9326082
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9393824
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9381566, upper bound: 0.9718202
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9381566, upper bound: 0.9718661
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9393804, upper bound: 0.9437539
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9381584
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9718184, upper bound: 0.9578369
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9253513, upper bound: 0.9437539
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.87
Output dim: 5, lower bound: -0.9577892, upper bound: 0.9381585
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.87
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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612831
time: 3.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674927
time: 5.52 seconds

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

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9612814
time: 5.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9680573
time: 3.55 seconds

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

Time for backsubstitution: 5.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9674926, upper bound: 0.9416506
time: 3.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9680553, upper bound: 0.9484246
time: 3.46 seconds

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

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9674928, upper bound: 0.9421956
time: 3.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9680555, upper bound: 0.9489707
time: 3.48 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1566582, -10.5510864, -13.3090010, -10.5028172, -2.0851469, 2.1364977
1: -11.2935362, -8.4465971, -11.4499931, -8.4285011, -2.4417810, 2.5665407
2: -10.6638451, -8.5526924, -10.7983446, -8.5783138, -2.0131693, 2.1840107
3: -4.4218569, -2.3489740, -4.5506554, -2.3368518, -1.9217591, 1.9366698
4: -15.1383734, -12.5253992, -15.1256409, -12.5049629, -2.1416450, 2.2526150
5: 8.2301731, 9.6845121, 8.2465858, 9.6683960, -1.2884603, 1.3015727
6: -4.7347169, -2.3376522, -4.7620850, -2.3217676, -2.0222750, 1.9797478
7: -15.7401857, -12.9499035, -15.7955828, -12.9915333, -2.3387957, 2.7112598
8: -0.7926233, 0.9181724, -0.7559106, 0.9562240, -1.4786463, 1.3487903
9: -6.6729708, -5.0150967, -6.6744580, -5.0117989, -1.6611719, 1.4439080

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9326082
time: 3.69 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9326082
time: 3.51 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1864090, -10.5393200, -13.3088646, -10.5032864, -2.1412148, 2.1477563
1: -11.3032475, -8.4618835, -11.4498463, -8.4335632, -2.5148249, 2.5596771
2: -10.7027826, -8.4952068, -10.7964144, -8.5783825, -2.0354395, 2.2764704
3: -4.4281187, -2.3455315, -4.5506287, -2.3369343, -1.9281893, 1.9398220
4: -15.1351528, -12.5331144, -15.1256332, -12.5064383, -2.1450338, 2.2534084
5: 8.2176399, 9.6887217, 8.2466173, 9.6668530, -1.2980504, 1.3143322
6: -4.7648597, -2.3143473, -4.7620201, -2.3228233, -2.0717940, 2.0016434
7: -15.7503281, -12.9482136, -15.7955198, -12.9926767, -2.3485003, 2.7101135
8: -0.8115044, 0.9446242, -0.7549405, 0.9562154, -1.5117426, 1.3710515
9: -6.6713901, -5.0119200, -6.6734362, -5.0118523, -1.6595378, 1.4665122

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9388181
time: 5.18 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9393824
time: 3.50 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1633673, -10.5471478, -13.3069401, -10.4902096, -2.1031342, 2.1629016
1: -11.3239450, -8.4196949, -11.4256086, -8.4348717, -2.4547157, 2.6014843
2: -10.7499352, -8.5666199, -10.7750120, -8.5727234, -2.1152086, 2.1644218
3: -4.3989763, -2.3425825, -4.5699735, -2.3410995, -1.9092627, 1.9532847
4: -15.1122828, -12.5838108, -15.1519194, -12.4279480, -2.1904612, 2.2176633
5: 8.2515154, 9.7071228, 8.2221375, 9.6680603, -1.2707794, 1.3486019
6: -4.7121363, -2.3152876, -4.7887893, -2.3123035, -2.0115733, 2.0380969
7: -15.7508335, -12.9453793, -15.7892513, -12.9876451, -2.3543100, 2.7017779
8: -0.7826340, 0.9031253, -0.7940555, 0.9715207, -1.4842033, 1.3754445
9: -6.6767054, -5.0664907, -6.6794038, -4.9594150, -1.7172904, 1.4160788

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9235567, upper bound: 0.9510088
time: 3.25 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9241194, upper bound: 0.9577828
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_B2_A2

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

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9235567, upper bound: 0.9319222
time: 3.45 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9241194, upper bound: 0.9386963
time: 3.58 seconds

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

Time for backsubstitution: 5.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
time: 3.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431905
time: 3.75 seconds

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

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9369782
time: 4.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9437539
time: 3.98 seconds

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

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9235586
time: 3.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577809, upper bound: 0.9241212
time: 3.98 seconds

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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510070, upper bound: 0.9241046
time: 3.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9577811, upper bound: 0.9246673
time: 3.70 seconds

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

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9369801
time: 3.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
time: 3.50 seconds

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

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9369801
time: 3.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9437539
time: 3.62 seconds

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

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9173474
time: 3.76 seconds

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

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9178934
time: 3.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9437521, upper bound: 0.9246672
time: 3.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.98 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612831
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674927
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9612814
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9490921, upper bound: 0.9680573
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9674926, upper bound: 0.9416506
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9680553, upper bound: 0.9484246
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9674928, upper bound: 0.9421956
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9680555, upper bound: 0.9489707
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9326082
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9326082
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9388181
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9369782, upper bound: 0.9393824
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9235567, upper bound: 0.9510088
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9241194, upper bound: 0.9577828
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9235567, upper bound: 0.9319222
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9241194, upper bound: 0.9386963
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431905
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9369782
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9388178, upper bound: 0.9437539
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9235586
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9577809, upper bound: 0.9241212
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9510070, upper bound: 0.9241046
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9577811, upper bound: 0.9246673
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9369801
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9431912
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9369801
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9247887, upper bound: 0.9437539
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9431892, upper bound: 0.9173474
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9437519, upper bound: 0.9241212
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.98
Output dim: 5, lower bound: -0.9431893, upper bound: 0.9178934
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.98
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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9418325
time: 5.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9614651
time: 4.79 seconds

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

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9478620
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9674946
time: 3.50 seconds

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

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9416505
time: 3.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9612830
time: 3.39 seconds

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

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9484248
time: 3.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9428806, upper bound: 0.9680574
time: 3.46 seconds

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

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9490937
time: 4.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9612811, upper bound: 0.9496567
time: 3.41 seconds

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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613272, upper bound: 0.9421965
time: 3.84 seconds

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613270, upper bound: 0.9484059
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613270, upper bound: 0.9489707
time: 3.47 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1566582, -10.5510864, -13.3041668, -10.5235815, -2.0469761, 2.1321707
1: -11.2935362, -8.4465971, -11.4438000, -8.4600401, -2.3868761, 2.5624423
2: -10.6638451, -8.5526924, -10.7377462, -8.5833111, -2.0108461, 2.1116185
3: -4.4218569, -2.3489740, -4.5487471, -2.3389950, -1.9192486, 1.9352195
4: -15.1383734, -12.5253992, -15.1254501, -12.5124207, -2.1314654, 2.2503557
5: 8.2301731, 9.6845121, 8.2474213, 9.6468496, -1.2653878, 1.2966213
6: -4.7347169, -2.3376522, -4.7603240, -2.3525896, -1.9811292, 1.9783614
7: -15.7401857, -12.9499035, -15.7909136, -13.0042801, -2.3259325, 2.7065845
8: -0.7926233, 0.9181724, -0.7319293, 0.9559593, -1.4753087, 1.3222166
9: -6.6729708, -5.0150967, -6.6658897, -5.0142002, -1.6587706, 1.4295340

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9175294, upper bound: 0.9326082
time: 3.61 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9175294, upper bound: 0.9326080
time: 3.79 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1566582, -10.5510864, -13.3339596, -10.5118170, -2.0820694, 2.1833563
1: -11.2935362, -8.4465971, -11.4536533, -8.4753218, -2.4330282, 2.6364765
2: -10.6638451, -8.5526924, -10.7766523, -8.5255909, -2.1047606, 2.1698010
3: -4.4218569, -2.3489740, -4.5550556, -2.3355613, -1.9227190, 1.9406986
4: -15.1383734, -12.5253992, -15.1222477, -12.5201187, -2.1324129, 2.2469873
5: 8.2301731, 9.6845121, 8.2348223, 9.6510172, -1.2779074, 1.3034432
6: -4.7347169, -2.3376522, -4.7910752, -2.3293247, -2.0205207, 2.0236661
7: -15.7401857, -12.9499035, -15.8011827, -13.0025578, -2.3303947, 2.7109456
8: -0.7926233, 0.9181724, -0.7507010, 0.9825034, -1.4951811, 1.3357449
9: -6.6729708, -5.0150967, -6.6642218, -5.0110049, -1.6619658, 1.4402835

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9175294, upper bound: 0.9326083
time: 3.64 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9175294, upper bound: 0.9326082
time: 3.48 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1864090, -10.5393200, -13.3041668, -10.5235815, -2.1029043, 2.1632311
1: -11.3032475, -8.4618835, -11.4438000, -8.4600401, -2.4603443, 2.6085806
2: -10.7027826, -8.4952068, -10.7377462, -8.5833111, -2.0688901, 2.2051404
3: -4.4281187, -2.3455315, -4.5487471, -2.3389950, -1.9257593, 1.9385414
4: -15.1351528, -12.5331144, -15.1254501, -12.5124207, -2.1361618, 2.2429304
5: 8.2176399, 9.6887217, 8.2474213, 9.6468496, -1.2756877, 1.3057704
6: -4.7648597, -2.3143473, -4.7603240, -2.3525896, -2.0304828, 2.0182931
7: -15.7503281, -12.9482136, -15.7909136, -13.0042801, -2.3359089, 2.7055202
8: -0.8115044, 0.9446242, -0.7319293, 0.9559593, -1.4859245, 1.3452382
9: -6.6713901, -5.0119200, -6.6658897, -5.0142002, -1.6571898, 1.4523575

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9173456, upper bound: 0.9388197
time: 3.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9173456, upper bound: 0.9388197
time: 3.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1864090, -10.5393200, -13.3339596, -10.5118170, -2.0800724, 2.1607840
1: -11.3032475, -8.4618835, -11.4536533, -8.4753218, -2.3887653, 2.5647445
2: -10.7027826, -8.4952068, -10.7766523, -8.5255909, -2.0500965, 2.1502936
3: -4.4281187, -2.3455315, -4.5550556, -2.3355613, -1.9270377, 1.9424953
4: -15.1351528, -12.5331144, -15.1222477, -12.5201187, -2.1337142, 2.2534328
5: 8.2176399, 9.6887217, 8.2348223, 9.6510172, -1.2832954, 1.3152260
6: -4.7648597, -2.3143473, -4.7910752, -2.3293247, -2.0031343, 2.0045240
7: -15.7503281, -12.9482136, -15.8011827, -13.0025578, -2.3417449, 2.7099710
8: -0.8115044, 0.9446242, -0.7507010, 0.9825034, -1.5112305, 1.3562745
9: -6.6713901, -5.0119200, -6.6642218, -5.0110049, -1.6603851, 1.4433708

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9173456, upper bound: 0.9393824
time: 3.65 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9173456, upper bound: 0.9393824
time: 3.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -13.1586208, -10.5684605, -13.3068628, -10.4907560, -2.0981708, 2.1280727
1: -11.3179722, -8.4517279, -11.4255104, -8.4353676, -2.4503012, 2.5454993
2: -10.6882601, -8.5720196, -10.7739153, -8.5727987, -2.0416594, 2.1613622
3: -4.3971014, -2.3447456, -4.5699434, -2.3411362, -1.9081850, 1.9513397
4: -15.1120892, -12.5913963, -15.1519175, -12.4280653, -2.1879616, 2.2115488
5: 8.2523460, 9.6852093, 8.2221508, 9.6677227, -1.2654440, 1.3272016
6: -4.7101393, -2.3466036, -4.7887592, -2.3128185, -2.0091429, 1.9968348
7: -15.7467308, -12.9583788, -15.7891741, -12.9878426, -2.3528490, 2.6901474
8: -0.7583771, 0.9028707, -0.7936804, 0.9715168, -1.4590893, 1.3720076
9: -6.6679935, -5.0688710, -6.6792703, -4.9594526, -1.7085409, 1.4140766

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9510080
time: 3.63 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9510086
time: 3.65 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -13.1882191, -10.5566940, -13.3067322, -10.4912262, -2.1541924, 2.1393316
1: -11.3276834, -8.4670172, -11.4253626, -8.4404325, -2.5233121, 2.5386369
2: -10.7271967, -8.5144558, -10.7719860, -8.5728626, -2.0639396, 2.2539334
3: -4.4033794, -2.3413148, -4.5699196, -2.3412189, -1.9146233, 1.9544897
4: -15.1088705, -12.5990896, -15.1519127, -12.4295397, -2.1913671, 2.2123508
5: 8.2397890, 9.6893950, 8.2221851, 9.6661768, -1.2750013, 1.3399388
6: -4.7403202, -2.3233070, -4.7886901, -2.3138757, -2.0587006, 2.0188720
7: -15.7568035, -12.9566870, -15.7891102, -12.9889879, -2.3625174, 2.6889987
8: -0.7772584, 0.9293244, -0.7927108, 0.9715083, -1.4921923, 1.3942730
9: -6.6664100, -5.0656857, -6.6782479, -4.9595084, -1.7069016, 1.4366899

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9572203
time: 3.53 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9577831
time: 3.56 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

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

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9319222
time: 3.89 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9319222
time: 3.75 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

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

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9381337
time: 3.46 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9370239, upper bound: 0.9386964
time: 3.79 seconds

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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9175295
time: 5.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9371619
time: 4.95 seconds

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

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9235564
time: 4.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9431894
time: 4.10 seconds

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

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9173474
time: 3.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9369801
time: 3.44 seconds

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

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9241213
time: 3.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9326063, upper bound: 0.9437539
time: 3.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.3068628, -10.4907560, -13.1586208, -10.5684605, -2.1280727, 2.0981708
1: -11.4255104, -8.4353676, -11.3179722, -8.4517279, -2.5454998, 2.4503014
2: -10.7739153, -8.5727987, -10.6882601, -8.5720196, -2.1613622, 2.0416594
3: -4.5699434, -2.3411362, -4.3971014, -2.3447456, -1.9513397, 1.9081850
4: -15.1519175, -12.4280653, -15.1120892, -12.5913963, -2.2115488, 2.1879618
5: 8.2221508, 9.6677227, 8.2523460, 9.6852093, -1.3272018, 1.2654440
6: -4.7887592, -2.3128185, -4.7101393, -2.3466036, -1.9968345, 2.0091431
7: -15.7891741, -12.9878426, -15.7467308, -12.9583788, -2.6901474, 2.3528490
8: -0.7936804, 0.9715168, -0.7583771, 0.9028707, -1.3720074, 1.4590895
9: -6.6792703, -4.9594526, -6.6679935, -5.0688710, -1.4140763, 1.7085409

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9185776
time: 4.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510068, upper bound: 0.9247882
time: 5.14 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.3067322, -10.4912262, -13.1882191, -10.5566940, -2.1393313, 2.1541927
1: -11.4253626, -8.4404325, -11.3276834, -8.4670172, -2.5386367, 2.5233116
2: -10.7719860, -8.5728626, -10.7271967, -8.5144558, -2.2539334, 2.0639396
3: -4.5699196, -2.3412189, -4.4033794, -2.3413148, -1.9544897, 1.9146233
4: -15.1519127, -12.4295397, -15.1088705, -12.5990896, -2.2123508, 2.1913667
5: 8.2221851, 9.6661768, 8.2397890, 9.6893950, -1.3399388, 1.2750013
6: -4.7886901, -2.3138757, -4.7403202, -2.3233070, -2.0188718, 2.0587006
7: -15.7891102, -12.9889879, -15.7568035, -12.9566870, -2.6889992, 2.3625178
8: -0.7927108, 0.9715083, -0.7772584, 0.9293244, -1.3942728, 1.4921925
9: -6.6782479, -4.9595084, -6.6664100, -5.0656857, -1.4366899, 1.7069016

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9572183, upper bound: 0.9185777
time: 5.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9572183, upper bound: 0.9253532
time: 3.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.3068628, -10.4907560, -13.1566315, -10.5564013, -2.1203675, 2.0878768
1: -11.4255104, -8.4353676, -11.2934866, -8.4585981, -2.5472212, 2.4434690
2: -10.7739153, -8.5727987, -10.6638260, -8.5663605, -2.1380820, 1.9900291
3: -4.5699434, -2.3411362, -4.4163847, -2.3490036, -1.9424820, 1.9067588
4: -15.1519175, -12.4280653, -15.1383724, -12.5146065, -2.2289267, 2.1771433
5: 8.2221508, 9.6677227, 8.2280121, 9.6844997, -1.3194153, 1.2786686
6: -4.7887592, -2.3128185, -4.7367697, -2.3376508, -1.9836669, 2.0087509
7: -15.7891741, -12.9878426, -15.7401762, -12.9545841, -2.7122021, 2.3682556
8: -0.7936804, 0.9715168, -0.7961533, 0.9181619, -1.3549984, 1.4600747
9: -6.6792703, -4.9594526, -6.6729279, -5.0165362, -1.4031019, 1.7134752

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510529, upper bound: 0.9178912
time: 4.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9510529, upper bound: 0.9241018
time: 5.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3067322, -10.4912262, -13.1863823, -10.5446339, -2.1316268, 2.1438968
1: -11.4253626, -8.4404325, -11.3031998, -8.4738846, -2.5403581, 2.5164776
2: -10.7719860, -8.5728626, -10.7027636, -8.5088758, -2.2306519, 2.0123050
3: -4.5699196, -2.3412189, -4.4226265, -2.3455589, -1.9456313, 1.9131963
4: -15.1519127, -12.4295397, -15.1351519, -12.5223141, -2.2297282, 2.1805477
5: 8.2221851, 9.6661768, 8.2154884, 9.6887102, -1.3321540, 1.2882246
6: -4.7886901, -2.3138757, -4.7669148, -2.3143487, -2.0057123, 2.0583067
7: -15.7891102, -12.9889879, -15.7503185, -12.9529018, -2.7110586, 2.3779252
8: -0.7927108, 0.9715083, -0.8150344, 0.9446130, -1.3772686, 1.4931800
9: -6.6782479, -4.9595084, -6.6713448, -5.0133581, -1.4257147, 1.7118363

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9572641, upper bound: 0.9178913
time: 5.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9572641, upper bound: 0.9246672
time: 3.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.3041668, -10.5235815, -13.3020706, -10.5062084, -2.0887408, 2.0802658
1: -11.4438000, -8.4600401, -11.4193640, -8.4549103, -2.4480901, 2.4270599
2: -10.7377462, -8.5833111, -10.7133293, -8.5639191, -2.0341516, 2.0118411
3: -4.5487471, -2.3389950, -4.5735259, -2.3432255, -1.7719598, 1.7891722
4: -15.1254501, -12.5124207, -15.1517277, -12.4463320, -2.1150079, 2.0735250
5: 8.2474213, 9.6468496, 8.2252398, 9.6461134, -1.2086989, 1.2318476
6: -4.7603240, -2.3525896, -4.7849193, -2.3436356, -1.9047768, 1.9172418
7: -15.7909136, -13.0042801, -15.7843103, -12.9958458, -2.3631420, 2.3440790
8: -0.7319293, 0.9559593, -0.7661648, 0.9712572, -1.3009965, 1.3186295
9: -6.6658897, -5.0142002, -6.6707277, -4.9604282, -1.4717743, 1.4419882

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9175313
time: 3.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9185776, upper bound: 0.9371639
time: 3.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.3339596, -10.5118170, -13.3020706, -10.5062084, -2.1400685, 2.1113284
1: -11.4536533, -8.4753218, -11.4193640, -8.4549103, -2.5218954, 2.4731975
2: -10.7766523, -8.5255909, -10.7133293, -8.5639191, -2.0922117, 2.1055841
3: -4.5550556, -2.3355613, -4.5735259, -2.3432255, -1.7773366, 1.7925026
4: -15.1222477, -12.5201187, -15.1517277, -12.4463320, -2.1200771, 2.0743980
5: 8.2348223, 9.6510172, 8.2252398, 9.6461134, -1.2198933, 1.2444069
6: -4.7910752, -2.3293247, -4.7849193, -2.3436356, -1.9500101, 1.9573076
7: -15.8011827, -13.0025578, -15.7843103, -12.9958458, -2.3730545, 2.3484249
8: -0.7507010, 0.9825034, -0.7661648, 0.9712572, -1.3147824, 1.3417956
9: -6.6642218, -5.0110049, -6.6707277, -4.9604282, -1.4825246, 1.4648376

Time for backsubstitution: 5.47 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.3380417823791504
rel_dist={5: [-1.0100506396824613, 1.010049981815225]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7611953, upper bound: 0.7527864
time: 3.85 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
time: 3.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.85 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 5, lower bound: -0.7611953, upper bound: 0.7527864
IS_B2, status: Status.UNKNOWN, split count: 1, time: 7.85
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -13.1630363, -10.4636698, -13.1613884, -10.5266781, -1.7224102, 1.7673178
1: -11.3005047, -8.3518839, -11.2995148, -8.4136467, -2.0236039, 2.0647445
2: -10.7255383, -8.5171347, -10.7255239, -8.5416927, -1.8423715, 1.8689811
3: -4.4308281, -2.2712052, -4.4290442, -2.3468359, -1.6718259, 1.7046008
4: -15.1814785, -12.4986553, -15.1385698, -12.5005589, -1.9163346, 1.8990476
5: 8.2192602, 9.7237635, 8.2217073, 9.7064924, -1.1107540, 1.1317589
6: -4.7438092, -2.2761712, -4.7417126, -2.3063393, -1.7244310, 1.7519121
7: -15.7447615, -12.9015751, -15.7444887, -12.9351664, -2.2579718, 2.3000317
8: -0.8540959, 0.9188125, -0.8229923, 0.9184313, -1.2392075, 1.1811650
9: -6.7087741, -5.0025487, -6.6816912, -5.0027695, -1.6319242, 1.6048036

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527863
time: 3.96 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
time: 3.54 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -13.1611977, -10.4716988, -13.3069639, -10.4817972, -1.7834792, 1.8661883
1: -11.2992468, -8.3757257, -11.4256668, -8.4219580, -2.0618978, 2.2326927
2: -10.7255001, -8.5286980, -10.7750311, -8.5532455, -1.8451490, 1.9428301
3: -4.4293098, -2.2901108, -4.5807705, -2.3410683, -1.6911969, 1.7237911
4: -15.1726646, -12.4999018, -15.1519251, -12.4215012, -1.8228168, 1.9043379
5: 8.2213888, 9.7083225, 8.2167139, 9.6680660, -1.0907633, 1.1852322
6: -4.7414055, -2.2867792, -4.7917042, -2.3123016, -1.7436256, 1.7334766
7: -15.7443609, -12.9263458, -15.7892628, -12.9812613, -1.9718962, 2.3794570
8: -0.8333356, 0.9182534, -0.7966328, 0.9715328, -1.3723292, 1.1516526
9: -6.7000766, -5.0027800, -6.6794314, -4.9480395, -1.6822462, 1.2938626

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7369448
time: 4.18 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7458996
time: 3.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.57 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 13.57
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527863
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 13.57
Output dim: 5, lower bound: -0.7527843, upper bound: 0.7527864
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 13.57
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7369448
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 13.57
Output dim: 5, lower bound: -0.7458978, upper bound: 0.7458996

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 5.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7453540, upper bound: 0.7458997
time: 3.40 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7543088, upper bound: 0.7458999
time: 3.92 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -13.3069639, -10.4817972, -13.1613884, -10.5266781, -1.8128591, 1.7485576
1: -11.4256668, -8.4219580, -11.2995148, -8.4136467, -2.2012644, 2.0459435
2: -10.7750311, -8.5532455, -10.7255239, -8.5416927, -1.9234843, 1.8449543
3: -4.5807705, -2.3410683, -4.4290442, -2.3468359, -1.6819391, 1.6690779
4: -15.1519251, -12.4215012, -15.1385698, -12.5005589, -1.8899870, 1.8005939
5: 8.2167139, 9.6680660, 8.2217073, 9.7064924, -1.1609821, 1.1108122
6: -4.7917042, -2.3123016, -4.7417126, -2.3063393, -1.7054849, 1.7198188
7: -15.7892628, -12.9812613, -15.7444887, -12.9351664, -2.3447485, 1.9669502
8: -0.7966328, 0.9715328, -0.8229923, 0.9184313, -1.1799982, 1.3140991
9: -6.6794314, -4.9480395, -6.6816912, -5.0027695, -1.2636719, 1.6564841

Time for backsubstitution: 5.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7453540, upper bound: 0.7458978
time: 5.62 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7543088, upper bound: 0.7458998
time: 3.53 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -13.1611786, -10.4783173, -13.3090773, -10.5022688, -1.7633343, 1.8458629
1: -11.2992344, -8.3776789, -11.4500942, -8.4280024, -2.0569215, 2.2478313
2: -10.7254963, -8.5403986, -10.7994423, -8.5782375, -1.7944870, 1.9037421
3: -4.4178658, -2.2901170, -4.5506849, -2.3368161, -1.6599455, 1.6901879
4: -15.1726608, -12.5363598, -15.1256447, -12.5048466, -1.7509017, 1.8537157
5: 8.2367191, 9.7083197, 8.2465706, 9.6687326, -1.0670211, 1.1482376
6: -4.7307720, -2.2867825, -4.7621164, -2.3212516, -1.7176256, 1.7050605
7: -15.7443619, -12.9298172, -15.7956581, -12.9913330, -1.9645290, 2.3848209
8: -0.8202765, 0.9182501, -0.7562866, 0.9562299, -1.3401914, 1.1110209
9: -6.7000608, -5.0240159, -6.6745906, -5.0117588, -1.6387825, 1.2389398

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226006
time: 3.98 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
time: 3.85 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -13.1611929, -10.4728317, -13.3069401, -10.4902096, -1.7641397, 1.8658710
1: -11.2992430, -8.3771009, -11.4256086, -8.4348717, -2.0642958, 2.2312284
2: -10.7254972, -8.5307751, -10.7750120, -8.5727234, -1.7732525, 1.9423304
3: -4.4277487, -2.2901125, -4.5699735, -2.3410995, -1.6901603, 1.6901891
4: -15.1726646, -12.5005894, -15.1519194, -12.4279480, -1.7669411, 1.9025340
5: 8.2220592, 9.7083225, 8.2221375, 9.6680603, -1.0896266, 1.1577001
6: -4.7410936, -2.2867780, -4.7887893, -2.3123035, -1.7435040, 1.7035388
7: -15.7443638, -12.9270325, -15.7892513, -12.9876451, -1.9880157, 2.3780894
8: -0.8330615, 0.9182515, -0.7940555, 0.9715207, -1.3722491, 1.1086762
9: -6.7000718, -5.0039930, -6.6794038, -4.9594150, -1.6670165, 1.2937958

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7315556
time: 4.15 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7370347
time: 6.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.32 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7453540, upper bound: 0.7458997
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7543088, upper bound: 0.7458999
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7453540, upper bound: 0.7458978
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7543088, upper bound: 0.7458998
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226006
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7315556
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.32
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7370347

## BFS IS instance: IS_B1_A1_A1

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

Time for backsubstitution: 5.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
time: 4.04 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7474756, upper bound: 0.7564734
time: 5.18 seconds

## BFS IS instance: IS_B1_A1_A2

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

Time for backsubstitution: 5.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511266, upper bound: 0.7549743
time: 3.99 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7564730, upper bound: 0.7564752
time: 3.82 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -13.3090773, -10.5022688, -13.1613722, -10.5332966, -1.7925382, 1.7284143
1: -11.4500942, -8.4280024, -11.2994995, -8.4156008, -2.2164021, 2.0409687
2: -10.7994423, -8.5782375, -10.7255211, -8.5533848, -1.8843951, 1.7942929
3: -4.5506849, -2.3368161, -4.4175997, -2.3468444, -1.6483355, 1.6378336
4: -15.1256447, -12.5048466, -15.1385651, -12.5370502, -1.8393650, 1.7286799
5: 8.2465706, 9.6687326, 8.2370262, 9.7064877, -1.1239877, 1.0871122
6: -4.7621164, -2.3212516, -4.7310743, -2.3063416, -1.6770692, 1.6938190
7: -15.7956581, -12.9913330, -15.7444887, -12.9386206, -2.3501148, 1.9595838
8: -0.7562866, 0.9562299, -0.8099329, 0.9184294, -1.1393790, 1.2819602
9: -6.6745906, -5.0117588, -6.6816764, -5.0240040, -1.2087469, 1.6130209

Time for backsubstitution: 5.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7310098, upper bound: 0.7354265
time: 4.20 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7364911, upper bound: 0.7370353
time: 5.80 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.1613827, -10.5278044, -1.8125415, 1.7292418
1: -11.4256086, -8.4348717, -11.2995081, -8.4150257, -2.1997995, 2.0483422
2: -10.7750120, -8.5727234, -10.7255230, -8.5437679, -1.9229832, 1.7730598
3: -4.5699735, -2.3410995, -4.4274831, -2.3468378, -1.6483285, 1.6680431
4: -15.1519194, -12.4279480, -15.1385689, -12.5012493, -1.8881831, 1.7447124
5: 8.2221375, 9.6680603, 8.2223759, 9.7064915, -1.1334507, 1.1096762
6: -4.7887893, -2.3123035, -4.7414012, -2.3063397, -1.6755276, 1.7196970
7: -15.7892513, -12.9876451, -15.7444878, -12.9358492, -2.3433819, 1.9830694
8: -0.7940555, 0.9715207, -0.8227153, 0.9184303, -1.1370249, 1.3140185
9: -6.6794038, -4.9594150, -6.6816854, -5.0039825, -1.2636061, 1.6412716

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7399646, upper bound: 0.7354276
time: 4.29 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7454459, upper bound: 0.7370370
time: 3.70 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -13.1564503, -10.4996300, -13.3077297, -10.5091209, -1.7486091, 1.8099210
1: -11.2932529, -8.4097118, -11.4483929, -8.4365149, -2.0363107, 2.1907768
2: -10.6638184, -8.5455027, -10.7807407, -8.5795660, -1.7204762, 1.8821707
3: -4.4160285, -2.2922707, -4.5501785, -2.3374286, -1.6582947, 1.6876843
4: -15.1724682, -12.5439243, -15.1255865, -12.5068340, -1.7456093, 1.8470058
5: 8.2375708, 9.6863480, 8.2468157, 9.6629982, -1.0545717, 1.1256752
6: -4.7287507, -2.3180981, -4.7616272, -2.3300266, -1.7048774, 1.6633210
7: -15.7400522, -12.9427843, -15.7944136, -12.9947395, -1.9596424, 2.3720613
8: -0.7959859, 0.9179902, -0.7498717, 0.9561501, -1.3137712, 1.0996653
9: -6.6913476, -5.0264087, -6.6723170, -5.0124073, -1.6278720, 1.2333124

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7225993
time: 5.83 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226006
time: 4.18 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1862116, -10.4878635, -13.3084154, -10.5053253, -1.8135676, 1.8195498
1: -11.3029757, -8.4250031, -11.4493256, -8.4452410, -2.1227250, 2.1873491
2: -10.7027540, -8.4880199, -10.7900562, -8.5786839, -1.7391405, 1.9891226
3: -4.4222956, -2.2888219, -4.5505123, -2.3371875, -1.6651154, 1.6910026
4: -15.1692486, -12.5516357, -15.1256161, -12.5097771, -1.7494226, 1.8498430
5: 8.2250328, 9.6905527, 8.2467136, 9.6629019, -1.0691534, 1.1374336
6: -4.7589302, -2.2947907, -4.7618217, -2.3261244, -1.7638307, 1.6778600
7: -15.7501926, -12.9410582, -15.7952347, -12.9955006, -1.9697275, 2.3718290
8: -0.8149047, 0.9444535, -0.7533553, 0.9561877, -1.3394341, 1.1275849
9: -6.6897783, -5.0232325, -6.6710076, -5.0120454, -1.6285062, 1.2585766

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
time: 4.00 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
time: 3.82 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.1564608, -10.4941425, -13.3056011, -10.4970589, -1.7493014, 1.8299093
1: -11.2932606, -8.4091339, -11.4239092, -8.4433861, -2.0437098, 2.1741672
2: -10.6638212, -8.5356846, -10.7563124, -8.5739956, -1.6992490, 1.9207745
3: -4.4259114, -2.2922635, -4.5694919, -2.3417053, -1.6885152, 1.6876721
4: -15.1724701, -12.5081749, -15.1518612, -12.4299374, -1.7616615, 1.8958273
5: 8.2229414, 9.6863518, 8.2223921, 9.6623058, -1.0771945, 1.1351948
6: -4.7390757, -2.3180959, -4.7882972, -2.3210754, -1.7307582, 1.6619174
7: -15.7400513, -12.9400091, -15.7879543, -12.9910526, -1.9832287, 2.3652611
8: -0.8087697, 0.9179935, -0.7876420, 0.9714403, -1.3458281, 1.0973290
9: -6.6913567, -5.0063877, -6.6771307, -4.9600654, -1.6561079, 1.2881117

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7315555
time: 3.64 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7315554
time: 3.92 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1862221, -10.4823771, -13.3062897, -10.4932652, -1.8142138, 1.8395536
1: -11.3029814, -8.4244261, -11.4248438, -8.4521084, -2.1300902, 2.1707458
2: -10.7027540, -8.4782753, -10.7656260, -8.5731544, -1.7179193, 2.0277205
3: -4.4321775, -2.2888167, -4.5698099, -2.3414660, -1.6953349, 1.6909876
4: -15.1692486, -12.5158854, -15.1518917, -12.4328842, -1.7654905, 1.8986640
5: 8.2104368, 9.6905565, 8.2222862, 9.6622219, -1.0918777, 1.1469332
6: -4.7692561, -2.2947881, -4.7884893, -2.3171744, -1.7897100, 1.6766057
7: -15.7501945, -12.9382925, -15.7888126, -12.9918098, -1.9932771, 2.3650737
8: -0.8276885, 0.9444551, -0.7911248, 0.9714806, -1.3714919, 1.1252584
9: -6.6897879, -5.0032105, -6.6758208, -4.9597025, -1.6567402, 1.3133821

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7370368
time: 3.94 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7370348
time: 5.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.86 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7421448, upper bound: 0.7549743
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7474756, upper bound: 0.7564734
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7511266, upper bound: 0.7549743
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7564730, upper bound: 0.7564752
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7310098, upper bound: 0.7354265
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7364911, upper bound: 0.7370353
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7399646, upper bound: 0.7354276
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7454459, upper bound: 0.7370370
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7225993
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7226006
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7280819
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7315555
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7354264, upper bound: 0.7315554
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7370368
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 14.86
Output dim: 5, lower bound: -0.7370351, upper bound: 0.7370348

## BFS IS instance: IS_B1_A1_A1_B1

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

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7382172, upper bound: 0.7372303
time: 5.15 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7382782, upper bound: 0.7511076
time: 4.31 seconds

## BFS IS instance: IS_B1_A1_A1_B2

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

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7435524, upper bound: 0.7387547
time: 4.16 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7436087, upper bound: 0.7526057
time: 5.49 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1600590, -10.5419407, -13.1566620, -10.5491152, -1.6802640, 1.6870418
1: -11.2978497, -8.4350758, -11.2935362, -8.4470577, -1.9643154, 2.0044639
2: -10.7068071, -8.5625324, -10.6638451, -8.5489597, -1.8191900, 1.6945634
3: -4.4176960, -2.3474734, -4.4256492, -2.3489730, -1.6255860, 1.6685345
4: -15.1385098, -12.5090113, -15.1383724, -12.5088367, -1.8889995, 1.8206379
5: 8.2273998, 9.7007198, 8.2232590, 9.6845121, -1.0543475, 1.0914981
6: -4.7382407, -2.3151054, -4.7393813, -2.3376501, -1.6424043, 1.7098279
7: -15.7433491, -12.9449978, -15.7401857, -12.9488564, -2.2362041, 2.2474055
8: -0.8140235, 0.9183445, -0.7984595, 0.9181726, -1.1191499, 1.1543109
9: -6.6793642, -5.0147805, -6.6729717, -5.0063729, -1.5918427, 1.5726275

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511265, upper bound: 0.7430712
time: 3.73 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7511267, upper bound: 0.7431058
time: 4.42 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1607265, -10.5381422, -13.1864109, -10.5373478, -1.6916251, 1.7519360
1: -11.2987289, -8.4438009, -11.3032465, -8.4623489, -1.9607658, 2.0908065
2: -10.7161217, -8.5616302, -10.7027807, -8.4914742, -1.9261642, 1.7131002
3: -4.4180040, -2.3472333, -4.4319134, -2.3455296, -1.6284714, 1.6753497
4: -15.1385374, -12.5119610, -15.1351528, -12.5165472, -1.8918529, 1.8143454
5: 8.2272949, 9.7006359, 8.2107601, 9.6887207, -1.0661392, 1.1007314
6: -4.7384491, -2.3112073, -4.7695251, -2.3143458, -1.6543736, 1.7687685
7: -15.7440968, -12.9457512, -15.7503290, -12.9471760, -2.2357635, 2.2499313
8: -0.8175254, 0.9183803, -0.8173418, 0.9446244, -1.1419446, 1.1802796
9: -6.6780562, -5.0144253, -6.6713905, -5.0031962, -1.6032176, 1.5732126

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7564729, upper bound: 0.7445783
time: 4.26 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7564731, upper bound: 0.7446142
time: 4.04 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3077297, -10.5091209, -13.1566496, -10.5546093, -1.7565827, 1.7136919
1: -11.4483929, -8.4365149, -11.2935276, -8.4476376, -2.1593490, 2.0203583
2: -10.7807407, -8.5795660, -10.6638432, -8.5587721, -1.8627141, 1.7201428
3: -4.5501785, -2.3374286, -4.4157672, -2.3489795, -1.6460419, 1.6361213
4: -15.1255865, -12.5068340, -15.1383734, -12.5446138, -1.8326812, 1.7233493
5: 8.2468157, 9.6629982, 8.2378788, 9.6845093, -1.1012478, 1.0747343
6: -4.7616272, -2.3300266, -4.7290587, -2.3376524, -1.6353352, 1.6810694
7: -15.7944136, -12.9947395, -15.7401867, -12.9516220, -2.3373156, 1.9547262
8: -0.7498717, 0.9561501, -0.7856748, 0.9181707, -1.1278801, 1.2559888
9: -6.6723170, -5.0124073, -6.6729627, -5.0263948, -1.2031221, 1.6021204

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7270232, upper bound: 0.7199176
time: 4.22 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7270233, upper bound: 0.7298831
time: 6.58 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3084154, -10.5053253, -13.1864004, -10.5428410, -1.7662249, 1.7786365
1: -11.4493256, -8.4452410, -11.3032408, -8.4629269, -2.1559215, 2.1067348
2: -10.7900562, -8.5786839, -10.7027807, -8.5012150, -1.9696908, 1.7388711
3: -4.5505123, -2.3371875, -4.4220285, -2.3455362, -1.6494279, 1.6429501
4: -15.1256161, -12.5097771, -15.1351528, -12.5523262, -1.8355136, 1.7270803
5: 8.2467136, 9.6629019, 8.2253466, 9.6887169, -1.1130687, 1.0892756
6: -4.7618217, -2.3261244, -4.7592015, -2.3143482, -1.6498895, 1.7400124
7: -15.7952347, -12.9955006, -15.7503281, -12.9499321, -2.3370171, 1.9647958
8: -0.7533553, 0.9561877, -0.8045552, 0.9446216, -1.1558146, 1.2819591
9: -6.6710076, -5.0120454, -6.6713810, -5.0232177, -1.2283857, 1.6027446

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7215243
time: 6.22 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7314940
time: 4.02 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3056011, -10.4970589, -13.1566620, -10.5491152, -1.7765672, 1.7144074
1: -11.4239092, -8.4433861, -11.2935362, -8.4470577, -2.1427383, 2.0277569
2: -10.7563124, -8.5739956, -10.6638451, -8.5489597, -1.9013166, 1.6989145
3: -4.5694919, -2.3417053, -4.4256492, -2.3489730, -1.6460214, 1.6663353
4: -15.1518612, -12.4299374, -15.1383724, -12.5088367, -1.8815022, 1.7393932
5: 8.2223921, 9.6623058, 8.2232590, 9.6845121, -1.1107678, 1.0973153
6: -4.7882972, -2.3210754, -4.7393813, -2.3376501, -1.6339111, 1.7069502
7: -15.7879543, -12.9910526, -15.7401857, -12.9488564, -2.3305168, 1.9783092
8: -0.7876420, 0.9714403, -0.7984595, 0.9181726, -1.1255341, 1.2880468
9: -6.6771307, -4.9600654, -6.6729717, -5.0063729, -1.2579253, 1.6303730

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7399645, upper bound: 0.7235155
time: 3.92 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7399647, upper bound: 0.7235462
time: 3.87 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3062897, -10.4932652, -13.1864109, -10.5373478, -1.7862239, 1.7793021
1: -11.4248438, -8.4521084, -11.3032465, -8.4623489, -2.1393175, 2.1140990
2: -10.7656260, -8.5731544, -10.7027807, -8.4914742, -2.0082889, 1.7176492
3: -4.5698099, -2.3414660, -4.4319134, -2.3455296, -1.6494050, 1.6731634
4: -15.1518917, -12.4328842, -15.1351528, -12.5165472, -1.8843336, 1.7431414
5: 8.2222862, 9.6622219, 8.2107601, 9.6887207, -1.1225686, 1.1119587
6: -4.7884893, -2.3171744, -4.7695251, -2.3143458, -1.6486149, 1.7658899
7: -15.7888126, -12.9918098, -15.7503290, -12.9471760, -2.3302612, 1.9883416
8: -0.7911248, 0.9714806, -0.8173418, 0.9446244, -1.1534798, 1.3140173
9: -6.6758208, -4.9597025, -6.6713905, -5.0031962, -1.2831948, 1.6309943

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7454458, upper bound: 0.7251224
time: 5.56 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7454460, upper bound: 0.7251548
time: 3.78 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -13.1566496, -10.5546093, -13.3077297, -10.5091209, -1.7136922, 1.7565830
1: -11.2935276, -8.4476376, -11.4483929, -8.4365149, -2.0203581, 2.1593487
2: -10.6638432, -8.5587721, -10.7807407, -8.5795660, -1.7201428, 1.8627138
3: -4.4157672, -2.3489795, -4.5501785, -2.3374286, -1.6361213, 1.6460421
4: -15.1383734, -12.5446138, -15.1255865, -12.5068340, -1.7233496, 1.8326812
5: 8.2378788, 9.6845093, 8.2468157, 9.6629982, -1.0747342, 1.1012478
6: -4.7290587, -2.3376524, -4.7616272, -2.3300266, -1.6810694, 1.6353354
7: -15.7401867, -12.9516220, -15.7944136, -12.9947395, -1.9547267, 2.3373160
8: -0.7856748, 0.9181707, -0.7498717, 0.9561501, -1.2559891, 1.1278802
9: -6.6729627, -5.0263948, -6.6723170, -5.0124073, -1.6021199, 1.2031224

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7196695, upper bound: 0.7170576
time: 3.95 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7298835, upper bound: 0.7170577
time: 4.19 seconds

## BFS IS instance: IS_B2_B1_A1_A2

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

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7196695, upper bound: 0.7170577
time: 3.69 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7298835, upper bound: 0.7170576
time: 4.15 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -13.1864004, -10.5428410, -13.3084154, -10.5053253, -1.7786365, 1.7662246
1: -11.3032408, -8.4629269, -11.4493256, -8.4452410, -2.1067348, 2.1559217
2: -10.7027807, -8.5012150, -10.7900562, -8.5786839, -1.7388711, 1.9696908
3: -4.4220285, -2.3455362, -4.5505123, -2.3371875, -1.6429505, 1.6494281
4: -15.1351528, -12.5523262, -15.1256161, -12.5097771, -1.7270808, 1.8355136
5: 8.2253466, 9.6887169, 8.2467136, 9.6629019, -1.0892756, 1.1130686
6: -4.7592015, -2.3143482, -4.7618217, -2.3261244, -1.7400126, 1.6498892
7: -15.7503281, -12.9499321, -15.7952347, -12.9955006, -1.9647961, 2.3370180
8: -0.8045552, 0.9446216, -0.7533553, 0.9561877, -1.2819591, 1.1558149
9: -6.6713810, -5.0232177, -6.6710076, -5.0120454, -1.6027446, 1.2283854

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7212782, upper bound: 0.7225389
time: 3.99 seconds

## Relational analysis of IS_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7314922, upper bound: 0.7225380
time: 5.36 seconds

## BFS IS instance: IS_B2_B1_A2_A2

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

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7212782, upper bound: 0.7225390
time: 3.82 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7314922, upper bound: 0.7225369
time: 4.13 seconds

## BFS IS instance: IS_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -13.1566620, -10.5491152, -13.3056011, -10.4970589, -1.7144070, 1.7765670
1: -11.2935362, -8.4470577, -11.4239092, -8.4433861, -2.0277576, 2.1427383
2: -10.6638451, -8.5489597, -10.7563124, -8.5739956, -1.6989145, 1.9013166
3: -4.4256492, -2.3489730, -4.5694919, -2.3417053, -1.6663356, 1.6460214
4: -15.1383724, -12.5088367, -15.1518612, -12.4299374, -1.7393932, 1.8815022
5: 8.2232590, 9.6845121, 8.2223921, 9.6623058, -1.0973151, 1.1107678
6: -4.7393813, -2.3376501, -4.7882972, -2.3210754, -1.7069502, 1.6339114
7: -15.7401857, -12.9488564, -15.7879543, -12.9910526, -1.9783087, 2.3305168
8: -0.7984595, 0.9181726, -0.7876420, 0.9714403, -1.2880468, 1.1255341
9: -6.6729717, -5.0063729, -6.6771307, -4.9600654, -1.6303730, 1.2579253

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7315554
time: 3.51 seconds

## Relational analysis of IS_B2_B2_A1_A1_A2

### Relational analysis result of IS_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7196734
time: 3.35 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -13.3020735, -10.5042353, -13.3056011, -10.4970589, -1.6796126, 1.6905341
1: -11.4193630, -8.4553709, -11.4239092, -8.4433861, -2.0413170, 2.0011656
2: -10.7133284, -8.5601854, -10.7563124, -8.5739956, -1.6938860, 1.8175931
3: -4.5773163, -2.3432238, -4.5694919, -2.3417053, -1.4840937, 1.4511330
4: -15.1517286, -12.4297695, -15.1518612, -12.4299374, -1.6657729, 1.7143176
5: 8.2182884, 9.6461153, 8.2223921, 9.6623058, -1.0664394, 1.0219378
6: -4.7895813, -2.3436337, -4.7882972, -2.3210754, -1.6101813, 1.5515120
7: -15.7843122, -12.9948969, -15.7879543, -12.9910526, -1.9850464, 1.9594183
8: -0.7719994, 0.9712582, -0.7876420, 0.9714403, -1.1204538, 1.0938818
9: -6.6707263, -4.9517045, -6.6771307, -4.9600654, -1.1915317, 1.2879834

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7315556
time: 3.34 seconds

## Relational analysis of IS_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7196715
time: 5.76 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -13.1864109, -10.5373478, -13.3062897, -10.4932652, -1.7793021, 1.7862244
1: -11.3032465, -8.4623489, -11.4248438, -8.4521084, -2.1140995, 2.1393178
2: -10.7027807, -8.4914742, -10.7656260, -8.5731544, -1.7176495, 2.0082884
3: -4.4319134, -2.3455296, -4.5698099, -2.3414660, -1.6731629, 1.6494048
4: -15.1351528, -12.5165472, -15.1518917, -12.4328842, -1.7431412, 1.8843336
5: 8.2107601, 9.6887207, 8.2222862, 9.6622219, -1.1119587, 1.1225686
6: -4.7695251, -2.3143458, -4.7884893, -2.3171744, -1.7658901, 1.6486149
7: -15.7503290, -12.9471760, -15.7888126, -12.9918098, -1.9883413, 2.3302617
8: -0.8173418, 0.9446244, -0.7911248, 0.9714806, -1.3140173, 1.1534798
9: -6.6713905, -5.0031962, -6.6758208, -4.9597025, -1.6309938, 1.2831945

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7370368
time: 4.20 seconds

## Relational analysis of IS_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7251527
time: 5.76 seconds

## BFS IS instance: IS_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -13.3320217, -10.4924698, -13.3062897, -10.4932652, -1.7399073, 1.7001395
1: -11.4292164, -8.4706497, -11.4248438, -8.4521084, -2.1280346, 1.9975998
2: -10.7522354, -8.5025482, -10.7656260, -8.5731544, -1.7125521, 1.9247041
3: -4.5836101, -2.3397784, -4.5698099, -2.3414660, -1.4895606, 1.4544787
4: -15.1485271, -12.4374828, -15.1518917, -12.4328842, -1.6698437, 1.7165873
5: 8.2057457, 9.6503086, 8.2222862, 9.6622219, -1.0820265, 1.0333052
6: -4.8202991, -2.3203626, -4.7884893, -2.3171744, -1.6652260, 1.5662270
7: -15.7946577, -12.9931812, -15.7888126, -12.9918098, -1.9951158, 1.9638467
8: -0.7907705, 0.9977992, -0.7911248, 0.9714806, -1.1453493, 1.1219290
9: -6.6690588, -4.9485116, -6.6758208, -4.9597025, -1.1951828, 1.3132706

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7370367
time: 4.00 seconds

## Relational analysis of IS_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7251548
time: 4.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.12 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7382172, upper bound: 0.7372303
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7382782, upper bound: 0.7511076
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7435524, upper bound: 0.7387547
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7436087, upper bound: 0.7526057
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7511265, upper bound: 0.7430712
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7511267, upper bound: 0.7431058
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7564729, upper bound: 0.7445783
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7564731, upper bound: 0.7446142
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7270232, upper bound: 0.7199176
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7270233, upper bound: 0.7298831
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7215243
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7314940
IS_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7399645, upper bound: 0.7235155
IS_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7399647, upper bound: 0.7235462
IS_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7454458, upper bound: 0.7251224
IS_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7454460, upper bound: 0.7251548
IS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7196695, upper bound: 0.7170576
IS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7298835, upper bound: 0.7170577
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7196695, upper bound: 0.7170577
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7298835, upper bound: 0.7170576
IS_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7212782, upper bound: 0.7225389
IS_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7314922, upper bound: 0.7225380
IS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7212782, upper bound: 0.7225390
IS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7314922, upper bound: 0.7225369
IS_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7315554
IS_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7196734
IS_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7315556
IS_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7196715
IS_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7370368
IS_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7251527
IS_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7370367
IS_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7251548

## BFS IS instance: IS_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.0905037, -10.5935860, -13.1380072, -10.5546169, -1.6142135, 1.6172974
1: -11.3226337, -8.4489784, -11.2916517, -8.4528503, -1.9458523, 1.9801848
2: -10.7261229, -8.5822554, -10.6623926, -8.5615120, -1.7591996, 1.6826081
3: -4.3993320, -2.3510079, -4.4150243, -2.3511095, -1.6320772, 1.6292782
4: -15.1069832, -12.5750074, -15.1370554, -12.5449238, -1.8282661, 1.8103862
5: 8.2621088, 9.6790428, 8.2379293, 9.6787682, -1.0327781, 1.0537882
6: -4.6443148, -2.3747063, -4.7120333, -2.3376689, -1.5909481, 1.6115024
7: -15.7330618, -12.9731693, -15.7384243, -12.9577389, -2.2129583, 2.2001371
8: -0.7743599, 0.8993974, -0.7851748, 0.9172549, -1.1206007, 1.1176938
9: -6.6708879, -5.0688038, -6.6720901, -5.0272284, -1.5774698, 1.5372524

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7382172, upper bound: 0.7334348
time: 6.07 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7382172, upper bound: 0.7372296
time: 5.40 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1417198, -10.5540562, -13.1518345, -10.5546255, -1.6131961, 1.6696968
1: -11.3186493, -8.4514837, -11.2926693, -8.4527664, -1.9726038, 1.9517336
2: -10.7269039, -8.5730724, -10.6628437, -8.5598373, -1.7528701, 1.7099195
3: -4.3983207, -2.3585515, -4.4157238, -2.3525553, -1.6286612, 1.6255336
4: -15.0944099, -12.5863419, -15.1344423, -12.5447369, -1.8178215, 1.8140445
5: 8.2522049, 9.6955051, 8.2379780, 9.6830769, -1.0362087, 1.0682313
6: -4.7009373, -2.3241267, -4.7257509, -2.3376689, -1.5647240, 1.6791205
7: -15.7493267, -12.9624958, -15.7400665, -12.9546404, -2.2400455, 2.2211514
8: -0.7741699, 0.8998182, -0.7851827, 0.9174581, -1.1209791, 1.1175950
9: -6.6679134, -5.0685229, -6.6715240, -5.0267239, -1.5785699, 1.5413394

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7382782, upper bound: 0.7473496
time: 4.20 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7382782, upper bound: 0.7511076
time: 4.16 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.0910950, -10.5897894, -13.1678190, -10.5428524, -1.6255112, 1.6822901
1: -11.3235188, -8.4577055, -11.3014107, -8.4681377, -1.9423409, 2.0666513
2: -10.7354412, -8.5813560, -10.7013226, -8.5037537, -1.8661861, 1.7011991
3: -4.3996658, -2.3507524, -4.4213266, -2.3476639, -1.6350079, 1.6361234
4: -15.1070099, -12.5779486, -15.1338367, -12.5526314, -1.8311419, 1.8041182
5: 8.2620049, 9.6789293, 8.2253971, 9.6829529, -1.0446377, 1.0629965
6: -4.6445246, -2.3708053, -4.7421885, -2.3143625, -1.6029105, 1.6704752
7: -15.7337885, -12.9739285, -15.7485542, -12.9560413, -2.2125578, 2.2026725
8: -0.7778351, 0.8994341, -0.8040261, 0.9437051, -1.1434014, 1.1436833
9: -6.6695786, -5.0684543, -6.6705084, -5.0240388, -1.5888991, 1.5378380

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7325072, upper bound: 0.7282345
time: 4.49 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7347013, upper bound: 0.7282345
time: 4.19 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1423702, -10.5502605, -13.1815891, -10.5428572, -1.6244404, 1.7346671
1: -11.3195295, -8.4602127, -11.3023767, -8.4680538, -1.9690466, 2.0381153
2: -10.7362251, -8.5722103, -10.7017937, -8.5023003, -1.8598981, 1.7283478
3: -4.3986473, -2.3582902, -4.4219894, -2.3491094, -1.6315637, 1.6324112
4: -15.0944366, -12.5892944, -15.1312170, -12.5524502, -1.8206687, 1.8077583
5: 8.2521067, 9.6954050, 8.2254486, 9.6872826, -1.0480293, 1.0773833
6: -4.7011499, -2.3202267, -4.7558756, -2.3143611, -1.5765944, 1.7380650
7: -15.7500420, -12.9632511, -15.7502089, -12.9529514, -2.2395668, 2.2237616
8: -0.7776957, 0.8998559, -0.8040841, 0.9439120, -1.1437721, 1.1435287
9: -6.6666050, -5.0681701, -6.6699414, -5.0235415, -1.5899520, 1.5419221

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7347532, upper bound: 0.7460766
time: 3.88 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7347532, upper bound: 0.7420851
time: 5.25 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.1600590, -10.5419407, -13.1586208, -10.5684605, -1.6610889, 1.6993551
1: -11.2978497, -8.4350758, -11.3179722, -8.4517279, -1.9607677, 2.0055916
2: -10.7068071, -8.5625324, -10.6882601, -8.5720196, -1.7680726, 1.7452619
3: -4.4176960, -2.3474734, -4.3971014, -2.3447456, -1.6486859, 1.6331038
4: -15.1385098, -12.5090113, -15.1120892, -12.5913963, -1.8151674, 1.8668365
5: 8.2273998, 9.7007198, 8.2523460, 9.6852093, -1.0704155, 1.0558828
6: -4.7382407, -2.3151054, -4.7101393, -2.3466036, -1.6659424, 1.6773095
7: -15.7433491, -12.9449978, -15.7467308, -12.9583788, -2.2241240, 2.2421069
8: -0.8140235, 0.9183445, -0.7583771, 0.9028707, -1.1510251, 1.1104211
9: -6.6793642, -5.0147805, -6.6679935, -5.0688710, -1.5498095, 1.5856853

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7471971, upper bound: 0.7253264
time: 4.23 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7472598, upper bound: 0.7392042
time: 4.22 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.1600590, -10.5419407, -13.1566315, -10.5564013, -1.6618800, 1.6870205
1: -11.2978497, -8.4350758, -11.2934866, -8.4585981, -1.9681416, 2.0044241
2: -10.7068071, -8.5625324, -10.6638260, -8.5663605, -1.7466893, 1.6944978
3: -4.4176960, -2.3474734, -4.4163847, -2.3490036, -1.6255255, 1.6264205
4: -15.1385098, -12.5090113, -15.1383724, -12.5146065, -1.8183870, 1.8206151
5: 8.2273998, 9.7007198, 8.2280121, 9.6844997, -1.0543404, 1.0653542
6: -4.7382407, -2.3151054, -4.7367697, -2.3376508, -1.6423798, 1.6719167
7: -15.7433491, -12.9449978, -15.7401762, -12.9545841, -2.2416191, 2.2473607
8: -0.8140235, 0.9183445, -0.7961533, 0.9181619, -1.1191213, 1.1044211
9: -6.6793642, -5.0147805, -6.6729279, -5.0165362, -1.5780592, 1.5725975

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7471972, upper bound: 0.7253634
time: 4.15 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7472600, upper bound: 0.7392388
time: 3.89 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.1607265, -10.5381422, -13.1882191, -10.5566940, -1.6724510, 1.7642527
1: -11.2987289, -8.4438009, -11.3276834, -8.4670172, -1.9572182, 2.0919352
2: -10.7161217, -8.5616302, -10.7271967, -8.5144558, -1.8751240, 1.7637639
3: -4.4180040, -2.3472333, -4.4033794, -2.3413148, -1.6515641, 1.6399260
4: -15.1385374, -12.5119610, -15.1088705, -12.5990896, -1.8180294, 1.8605397
5: 8.2272949, 9.7006359, 8.2397890, 9.6893950, -1.0822104, 1.0651149
6: -4.7384491, -2.3112073, -4.7403202, -2.3233070, -1.6779099, 1.7362881
7: -15.7440968, -12.9457512, -15.7568035, -12.9566870, -2.2236834, 2.2446342
8: -0.8175254, 0.9183803, -0.7772584, 0.9293244, -1.1738205, 1.1363978
9: -6.6780562, -5.0144253, -6.6664100, -5.0656857, -1.5611911, 1.5862703

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7525497, upper bound: 0.7268588
time: 3.76 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7526058, upper bound: 0.7407111
time: 3.83 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.1607265, -10.5381422, -13.1863823, -10.5446339, -1.6732531, 1.7519159
1: -11.2987289, -8.4438009, -11.3031998, -8.4738846, -1.9645853, 2.0907664
2: -10.7161217, -8.5616302, -10.7027636, -8.5088758, -1.8537393, 1.7130342
3: -4.4180040, -2.3472333, -4.4226265, -2.3455589, -1.6284122, 1.6332438
4: -15.1385374, -12.5119610, -15.1351519, -12.5223141, -1.8212509, 1.8143222
5: 8.2272949, 9.7006359, 8.2154884, 9.6887102, -1.0661325, 1.0745738
6: -4.7384491, -2.3112073, -4.7669148, -2.3143487, -1.6543493, 1.7308950
7: -15.7440968, -12.9457512, -15.7503185, -12.9529018, -2.2411375, 2.2498875
8: -0.8175254, 0.9183803, -0.8150344, 0.9446130, -1.1419163, 1.1304002
9: -6.6780562, -5.0144253, -6.6713448, -5.0133581, -1.5894399, 1.5731816

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7525499, upper bound: 0.7268949
time: 4.03 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7526060, upper bound: 0.7407445
time: 6.46 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.2366257, -10.5431509, -13.1380072, -10.5546169, -1.6928723, 1.6486781
1: -11.4510231, -8.4492998, -11.2916517, -8.4528503, -2.1503096, 2.0081623
2: -10.7750978, -8.5922222, -10.6623926, -8.5615120, -1.8444219, 1.6884656
3: -4.5556893, -2.3425195, -4.4150243, -2.3511095, -1.6477594, 1.6299157
4: -15.1203403, -12.4943094, -15.1370554, -12.5449238, -1.8251328, 1.7187803
5: 8.2545815, 9.6417236, 8.2379293, 9.6787682, -1.0913650, 1.0572984
6: -4.7039561, -2.3759394, -4.7120333, -2.3376689, -1.5842576, 1.6203427
7: -15.7886982, -13.0111666, -15.7384243, -12.9577389, -2.3231688, 1.9363949
8: -0.7481709, 0.9524682, -0.7851748, 0.9172549, -1.1241736, 1.2508559
9: -6.6692920, -5.0134568, -6.6720901, -5.0272284, -1.1997917, 1.5956340

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7270232, upper bound: 0.7161324
time: 3.55 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7270232, upper bound: 0.7199175
time: 4.16 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2835503, -10.5091743, -13.1518345, -10.5546255, -1.6922626, 1.6987224
1: -11.4447460, -8.4669867, -11.2926693, -8.4527664, -2.1508093, 1.9764843
2: -10.7768955, -8.5851173, -10.6628437, -8.5598373, -1.8391652, 1.7104936
3: -4.5500097, -2.3560867, -4.4157238, -2.3525553, -1.6420312, 1.6243277
4: -15.1077490, -12.5073652, -15.1344423, -12.5447369, -1.8143487, 1.7212822
5: 8.2472630, 9.6557732, 8.2379780, 9.6830769, -1.0929099, 1.0730895
6: -4.7490950, -2.3300939, -4.7257509, -2.3376689, -1.5514247, 1.6762350
7: -15.7939205, -13.0155239, -15.7400665, -12.9546404, -2.3343201, 1.9321227
8: -0.7479472, 0.9529533, -0.7851827, 0.9174581, -1.1248205, 1.2514944
9: -6.6654334, -5.0138059, -6.6715240, -5.0267239, -1.1955814, 1.5988674

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_B1_A2_A1_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7270233, upper bound: 0.7261002
time: 3.78 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7270233, upper bound: 0.7298831
time: 6.60 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.2372475, -10.5393534, -13.1678190, -10.5428524, -1.7024560, 1.7136714
1: -11.4519653, -8.4580231, -11.3014107, -8.4681377, -2.1468611, 2.0946283
2: -10.7844210, -8.5913696, -10.7013226, -8.5037537, -1.9514079, 1.7072225
3: -4.5560274, -2.3422647, -4.4213266, -2.3476639, -1.6511502, 1.6367722
4: -15.1203671, -12.4972553, -15.1338367, -12.5526314, -1.8279872, 1.7224944
5: 8.2544737, 9.6416092, 8.2253971, 9.6829529, -1.1032310, 1.0718350
6: -4.7041469, -2.3720412, -4.7421885, -2.3143625, -1.5988050, 1.6793008
7: -15.7895012, -13.0119267, -15.7485542, -12.9560413, -2.3228941, 1.9465008
8: -0.7516272, 0.9525075, -0.8040261, 0.9437051, -1.1521435, 1.2768455
9: -6.6679811, -5.0131021, -6.6705084, -5.0240388, -1.2250779, 1.5962563

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7096116
time: 5.68 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7215243
time: 5.40 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2842331, -10.5053806, -13.1815891, -10.5428572, -1.7018557, 1.7636924
1: -11.4456778, -8.4757099, -11.3023767, -8.4680538, -2.1473794, 2.0628653
2: -10.7862196, -8.5843239, -10.7017937, -8.5023003, -1.9461889, 1.7291031
3: -4.5503464, -2.3558271, -4.4219894, -2.3491094, -1.6454253, 1.6312175
4: -15.1077776, -12.5103130, -15.1312170, -12.5524502, -1.8171740, 1.7249973
5: 8.2471609, 9.6556711, 8.2254486, 9.6872826, -1.1047400, 1.0875767
6: -4.7492976, -2.3261914, -4.7558756, -2.3143611, -1.5659375, 1.7351794
7: -15.7947454, -13.0162849, -15.7502089, -12.9529514, -2.3340244, 1.9422128
8: -0.7514534, 0.9529910, -0.8040841, 0.9439120, -1.1527627, 1.2774286
9: -6.6641226, -5.0134439, -6.6699414, -5.0235415, -1.2208328, 1.5994849

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7236403, upper bound: 0.7250156
time: 5.12 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7236403, upper bound: 0.7210576
time: 5.85 seconds

## BFS IS instance: IS_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.3056011, -10.4970589, -13.1586208, -10.5684605, -1.7528343, 1.7267220
1: -11.4239092, -8.4433861, -11.3179722, -8.4517279, -2.1391892, 2.0288856
2: -10.7563124, -8.5739956, -10.6882601, -8.5720196, -1.8501992, 1.7486944
3: -4.5694919, -2.3417053, -4.3971014, -2.3447456, -1.6607356, 1.6309047
4: -15.1518612, -12.4299374, -15.1120892, -12.5913963, -1.8076701, 1.7696633
5: 8.2223921, 9.6623058, 8.2523460, 9.6852093, -1.1268704, 1.0599589
6: -4.7882972, -2.3210754, -4.7101393, -2.3466036, -1.6523931, 1.6744316
7: -15.7879543, -12.9910526, -15.7467308, -12.9583788, -2.3184366, 1.9687810
8: -0.7876420, 0.9714403, -0.7583771, 0.9028707, -1.1510651, 1.2441571
9: -6.6771307, -4.9600654, -6.6679935, -5.0688710, -1.1878953, 1.6434331

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7359779, upper bound: 0.7080048
time: 3.92 seconds

## Relational analysis of IS_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7359780, upper bound: 0.7179725
time: 4.58 seconds

## BFS IS instance: IS_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.3056011, -10.4970589, -13.1566315, -10.5564013, -1.7443080, 1.7143860
1: -11.4239092, -8.4433861, -11.2934866, -8.4585981, -2.1465826, 2.0277171
2: -10.7563124, -8.5739956, -10.6638260, -8.5663605, -1.8288169, 1.6988525
3: -4.5694919, -2.3417053, -4.4163847, -2.3490036, -1.6459866, 1.6242666
4: -15.1518612, -12.4299374, -15.1383724, -12.5146065, -1.8109112, 1.7393916
5: 8.2223921, 9.6623058, 8.2280121, 9.6844997, -1.1107609, 1.0653967
6: -4.7882972, -2.3210754, -4.7367697, -2.3376508, -1.6339107, 1.6690354
7: -15.7879543, -12.9910526, -15.7401762, -12.9545841, -2.3359985, 1.9783030
8: -0.7876420, 0.9714403, -0.7961533, 0.9181619, -1.1255219, 1.2381580
9: -6.6771307, -4.9600654, -6.6729279, -5.0165362, -1.1709394, 1.6303425

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7359781, upper bound: 0.7080354
time: 3.48 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7359782, upper bound: 0.7180032
time: 4.15 seconds

## BFS IS instance: IS_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.3062897, -10.4932652, -13.1882191, -10.5566940, -1.7624917, 1.7916193
1: -11.4248438, -8.4521084, -11.3276834, -8.4670172, -2.1357675, 2.1152272
2: -10.7656260, -8.5731544, -10.7271967, -8.5144558, -1.9572487, 1.7673912
3: -4.5698099, -2.3414660, -4.4033794, -2.3413148, -1.6641054, 1.6377397
4: -15.1518917, -12.4328842, -15.1088705, -12.5990896, -1.8105106, 1.7734115
5: 8.2222862, 9.6622219, 8.2397890, 9.6893950, -1.1386734, 1.0745504
6: -4.7884893, -2.3171744, -4.7403202, -2.3233070, -1.6670985, 1.7334094
7: -15.7888126, -12.9918098, -15.7568035, -12.9566870, -2.3181801, 1.9788132
8: -0.7911248, 0.9714806, -0.7772584, 0.9293244, -1.1790214, 1.2701354
9: -6.6758208, -4.9597025, -6.6664100, -5.0656857, -1.2131696, 1.6440535

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A2_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7414593, upper bound: 0.7096133
time: 4.02 seconds

## Relational analysis of IS_B1_A2_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7414593, upper bound: 0.7195812
time: 4.61 seconds

## BFS IS instance: IS_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3062897, -10.4932652, -13.1863823, -10.5446339, -1.7539489, 1.7792821
1: -11.4248438, -8.4521084, -11.3031998, -8.4738846, -2.1431541, 2.1140590
2: -10.7656260, -8.5731544, -10.7027636, -8.5088758, -1.9358640, 1.7175875
3: -4.5698099, -2.3414660, -4.4226265, -2.3455589, -1.6493700, 1.6311023
4: -15.1518917, -12.4328842, -15.1351519, -12.5223141, -1.8137531, 1.7431386
5: 8.2222862, 9.6622219, 8.2154884, 9.6887102, -1.1225619, 1.0799602
6: -4.7884893, -2.3171744, -4.7669148, -2.3143487, -1.6486137, 1.7280135
7: -15.7888126, -12.9918098, -15.7503185, -12.9529018, -2.3356991, 1.9883358
8: -0.7911248, 0.9714806, -0.8150344, 0.9446130, -1.1534672, 1.2641377
9: -6.6758208, -4.9597025, -6.6713448, -5.0133581, -1.1962121, 1.6309628

Time for backsubstitution: 5.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_B1_A2_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7414594, upper bound: 0.7096441
time: 3.64 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7414595, upper bound: 0.7196119
time: 3.93 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1380072, -10.5546169, -13.2366257, -10.5431509, -1.6486778, 1.6928725
1: -11.2916517, -8.4528503, -11.4510231, -8.4492998, -2.0081615, 2.1503096
2: -10.6623926, -8.5615120, -10.7750978, -8.5922222, -1.6884661, 1.8444216
3: -4.4150243, -2.3511095, -4.5556893, -2.3425195, -1.6299162, 1.6477592
4: -15.1370554, -12.5449238, -15.1203403, -12.4943094, -1.7187800, 1.8251328
5: 8.2379293, 9.6787682, 8.2545815, 9.6417236, -1.0572984, 1.0913651
6: -4.7120333, -2.3376689, -4.7039561, -2.3759394, -1.6203425, 1.5842577
7: -15.7384243, -12.9577389, -15.7886982, -13.0111666, -1.9363956, 2.3231688
8: -0.7851748, 0.9172549, -0.7481709, 0.9524682, -1.2508559, 1.1241736
9: -6.6720901, -5.0272284, -6.6692920, -5.0134568, -1.5956345, 1.1997917

Time for backsubstitution: 5.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B2_B1_A1_A1_B1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7161304, upper bound: 0.7270251
time: 4.24 seconds

## Relational analysis of IS_B2_B1_A1_A1_B1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7161304, upper bound: 0.7270251
time: 3.64 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1518345, -10.5546255, -13.2835503, -10.5091743, -1.6987224, 1.6922626
1: -11.2926693, -8.4527664, -11.4447460, -8.4669867, -1.9764848, 2.1508093
2: -10.6628437, -8.5598373, -10.7768955, -8.5851173, -1.7104936, 1.8391654
3: -4.4157238, -2.3525553, -4.5500097, -2.3560867, -1.6243281, 1.6420312
4: -15.1344423, -12.5447369, -15.1077490, -12.5073652, -1.7212820, 1.8143489
5: 8.2379780, 9.6830769, 8.2472630, 9.6557732, -1.0730896, 1.0929098
6: -4.7257509, -2.3376689, -4.7490950, -2.3300939, -1.6762347, 1.5514247
7: -15.7400665, -12.9546404, -15.7939205, -13.0155239, -1.9321232, 2.3343201
8: -0.7851827, 0.9174581, -0.7479472, 0.9529533, -1.2514942, 1.1248202
9: -6.6715240, -5.0267239, -6.6654334, -5.0138059, -1.5988669, 1.1955812

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7270252
time: 3.66 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7270250
time: 3.90 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -13.2842102, -10.5097361, -13.2366257, -10.5431509, -1.6136112, 1.6154733
1: -11.4173880, -8.4591579, -11.4510231, -8.4492998, -2.0174055, 1.9832942
2: -10.7117386, -8.5724773, -10.7750978, -8.5922222, -1.6827009, 1.7565300
3: -4.5666757, -2.3446670, -4.5556893, -2.3425195, -1.4531307, 1.4468913
4: -15.1504116, -12.4659719, -15.1203403, -12.4943094, -1.6262119, 1.6545672
5: 8.2329969, 9.6407719, 8.2545815, 9.6417236, -1.0237570, 1.0027544
6: -4.7640686, -2.3436515, -4.7039561, -2.3759394, -1.5070884, 1.4984269
7: -15.7823076, -13.0017033, -15.7886982, -13.0111666, -1.9354134, 1.9388678
8: -0.7587276, 0.9703326, -0.7481709, 0.9524682, -1.0851986, 1.0920137
9: -6.6699648, -4.9725780, -6.6692920, -5.0134568, -1.2049325, 1.2301736

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7158843, upper bound: 0.7170576
time: 3.59 seconds

## Relational analysis of IS_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7158843, upper bound: 0.7170577
time: 4.45 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -13.2966156, -10.5097427, -13.2835503, -10.5091743, -1.6691551, 1.6047549
1: -11.4185057, -8.4626637, -11.4447460, -8.4669867, -1.9848766, 2.0095522
2: -10.7124386, -8.5712252, -10.7768955, -8.5851173, -1.7068076, 1.7508292
3: -4.5673990, -2.3474231, -4.5500097, -2.3560867, -1.4426439, 1.4470139
4: -15.1477947, -12.4657879, -15.1077490, -12.5073652, -1.6476176, 1.6292119
5: 8.2330456, 9.6443806, 8.2472630, 9.6557732, -1.0348651, 1.0068933
6: -4.7755556, -2.3436527, -4.7490950, -2.3300939, -1.5834014, 1.4660301
7: -15.7841797, -13.0021381, -15.7939205, -13.0155239, -1.9470277, 1.9685137
8: -0.7587633, 0.9705529, -0.7479472, 0.9529533, -1.0872507, 1.0924656
9: -6.6692071, -4.9720526, -6.6654334, -5.0138059, -1.2058201, 1.2262721

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7170577
time: 3.63 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7170577
time: 3.72 seconds

## BFS IS instance: IS_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1678190, -10.5428524, -13.2372475, -10.5393534, -1.7136722, 1.7024562
1: -11.3014107, -8.4681377, -11.4519653, -8.4580231, -2.0946283, 2.1468608
2: -10.7013226, -8.5037537, -10.7844210, -8.5913696, -1.7072225, 1.9514077
3: -4.4213266, -2.3476639, -4.5560274, -2.3422647, -1.6367722, 1.6511500
4: -15.1338367, -12.5526314, -15.1203671, -12.4972553, -1.7224941, 1.8279870
5: 8.2253971, 9.6829529, 8.2544737, 9.6416092, -1.0718350, 1.1032310
6: -4.7421885, -2.3143625, -4.7041469, -2.3720412, -1.6793008, 1.5988051
7: -15.7485542, -12.9560413, -15.7895012, -13.0119267, -1.9465003, 2.3228946
8: -0.8040261, 0.9437051, -0.7516272, 0.9525075, -1.2768452, 1.1521435
9: -6.6705084, -5.0240388, -6.6679811, -5.0131021, -1.5962563, 1.2250776

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7096113, upper bound: 0.7325048
time: 6.01 seconds

## Relational analysis of IS_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7096113, upper bound: 0.7325063
time: 3.90 seconds

## BFS IS instance: IS_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1815891, -10.5428572, -13.2842331, -10.5053806, -1.7636924, 1.7018557
1: -11.3023767, -8.4680538, -11.4456778, -8.4757099, -2.0628653, 2.1473794
2: -10.7017937, -8.5023003, -10.7862196, -8.5843239, -1.7291031, 1.9461889
3: -4.4219894, -2.3491094, -4.5503464, -2.3558271, -1.6312170, 1.6454253
4: -15.1312170, -12.5524502, -15.1077776, -12.5103130, -1.7249975, 1.8171740
5: 8.2254486, 9.6872826, 8.2471609, 9.6556711, -1.0875769, 1.1047399
6: -4.7558756, -2.3143611, -4.7492976, -2.3261914, -1.7351794, 1.5659373
7: -15.7502089, -12.9529514, -15.7947454, -13.0162849, -1.9422126, 2.3340249
8: -0.8040841, 0.9439120, -0.7514534, 0.9529910, -1.2774284, 1.1527628
9: -6.6699414, -5.0235415, -6.6641226, -5.0134439, -1.5994849, 1.2208323

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7250155, upper bound: 0.7236419
time: 4.51 seconds

## Relational analysis of IS_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7210573, upper bound: 0.7236420
time: 3.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.18 seconds
IS_B1_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7382172, upper bound: 0.7334348
IS_B1_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7382172, upper bound: 0.7372296
IS_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7382782, upper bound: 0.7473496
IS_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7382782, upper bound: 0.7511076
IS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7325072, upper bound: 0.7282345
IS_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7347013, upper bound: 0.7282345
IS_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7347532, upper bound: 0.7460766
IS_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7347532, upper bound: 0.7420851
IS_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7471971, upper bound: 0.7253264
IS_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7472598, upper bound: 0.7392042
IS_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7471972, upper bound: 0.7253634
IS_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7472600, upper bound: 0.7392388
IS_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7525497, upper bound: 0.7268588
IS_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7526058, upper bound: 0.7407111
IS_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7525499, upper bound: 0.7268949
IS_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7526060, upper bound: 0.7407445
IS_B1_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7270232, upper bound: 0.7161324
IS_B1_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7270232, upper bound: 0.7199175
IS_B1_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7270233, upper bound: 0.7261002
IS_B1_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7270233, upper bound: 0.7298831
IS_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7096116
IS_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7325046, upper bound: 0.7215243
IS_B1_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7236403, upper bound: 0.7250156
IS_B1_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7236403, upper bound: 0.7210576
IS_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7359779, upper bound: 0.7080048
IS_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7359780, upper bound: 0.7179725
IS_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7359781, upper bound: 0.7080354
IS_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7359782, upper bound: 0.7180032
IS_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7414593, upper bound: 0.7096133
IS_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7414593, upper bound: 0.7195812
IS_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7414594, upper bound: 0.7096441
IS_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7414595, upper bound: 0.7196119
IS_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7161304, upper bound: 0.7270251
IS_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7161304, upper bound: 0.7270251
IS_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7270252
IS_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7270250
IS_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7158843, upper bound: 0.7170576
IS_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7158843, upper bound: 0.7170577
IS_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7170577
IS_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7260983, upper bound: 0.7170577
IS_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7096113, upper bound: 0.7325048
IS_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7096113, upper bound: 0.7325063
IS_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7250155, upper bound: 0.7236419
IS_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 14.18
Output dim: 5, lower bound: -0.7210573, upper bound: 0.7236420
IS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7212782, upper bound: 0.7225390
IS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7314922, upper bound: 0.7225369
IS_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7315554
IS_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7196734
IS_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7315556
IS_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7235135, upper bound: 0.7196715
IS_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7370368
IS_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7251527
IS_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7370367
IS_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 14.18
Output dim: 5, lower bound: -0.7251221, upper bound: 0.7251548
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.139000654220581
rel_dist={5: [-0.78768892317715, 0.7876895182364123]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2375
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 2375

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6815778
time: 4.19 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749160
time: 3.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.78
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6815778
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.78
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

Time for backsubstitution: 5.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749159
time: 3.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749160
time: 3.44 seconds

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

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6626736, upper bound: 0.6693756
time: 4.09 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693756
time: 4.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749159
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 5, lower bound: -0.6749140, upper bound: 0.6749160
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 5, lower bound: -0.6626736, upper bound: 0.6693756
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 13.90
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693756

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.1613884, -10.5266781, -1.6006651, 1.6006651
1: -11.2995148, -8.4136467, -11.2995148, -8.4136467, -1.8875680, 1.8875678
2: -10.7255239, -8.5416927, -10.7255239, -8.5416927, -1.7439160, 1.7439158
3: -4.4290442, -2.3468359, -4.4290442, -2.3468359, -1.5789943, 1.5789943
4: -15.1385698, -12.5005589, -15.1385698, -12.5005589, -1.7630677, 1.7630677
5: 8.2217073, 9.7064924, 8.2217073, 9.7064924, -1.0381987, 1.0381988
6: -4.7417126, -2.3063393, -4.7417126, -2.3063393, -1.6145558, 1.6145556
7: -15.7444887, -12.9351664, -15.7444887, -12.9351664, -2.1266613, 2.1266618
8: -0.8229923, 0.9184313, -0.8229923, 0.9184313, -1.1090004, 1.1090004
9: -6.6816912, -5.0027695, -6.6816912, -5.0027695, -1.5272102, 1.5272102

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6626736, upper bound: 0.6760375
time: 4.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6760357
time: 5.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -13.1613884, -10.5266781, -13.3069639, -10.4817972, -1.6280313, 1.6881618
1: -11.2995148, -8.4136467, -11.4256668, -8.4219580, -1.9108620, 2.0661862
2: -10.7255239, -8.5416927, -10.7750311, -8.5532455, -1.7474780, 1.8260374
3: -4.4290442, -2.3468359, -4.5807705, -2.3410683, -1.5768671, 1.5851855
4: -15.1385698, -12.5005589, -15.1519251, -12.4215012, -1.6621017, 1.7555509
5: 8.2217073, 9.7064924, 8.2167139, 9.6680660, -1.0446699, 1.0946354
6: -4.7417126, -2.3063393, -4.7917042, -2.3123016, -1.6116800, 1.5907993
7: -15.7444887, -12.9351664, -15.7892628, -12.9812613, -1.8400259, 2.2212563
8: -0.8229923, 0.9184313, -0.7966328, 0.9715328, -1.2427392, 1.1090474
9: -6.6816912, -5.0027695, -6.6794314, -4.9480395, -1.5850139, 1.1894748

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693374
time: 5.14 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6760373
time: 3.83 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -13.3090773, -10.5022688, -13.1606321, -10.4813862, -1.7173219, 1.6288691
1: -11.4500942, -8.4280024, -11.2988634, -8.3856258, -2.1107492, 1.9126277
2: -10.7994423, -8.5782375, -10.7254848, -8.5455933, -1.8003397, 1.6944993
3: -4.5506849, -2.3368161, -4.4145603, -2.2959781, -1.5923753, 1.5539107
4: -15.1256447, -12.5048466, -15.1697903, -12.5454369, -1.7067790, 1.6118917
5: 8.2465706, 9.6687326, 8.2409697, 9.7037106, -1.0817361, 0.9967308
6: -4.7621164, -2.3212516, -4.7274046, -2.2900825, -1.5882952, 1.6011081
7: -15.7956581, -12.9913330, -15.7442513, -12.9380465, -2.2563715, 1.8337467
8: -0.7562866, 0.9562299, -0.8114667, 0.9180830, -1.0362949, 1.2646911
9: -6.6745906, -5.0117588, -6.6974535, -5.0293727, -1.1537964, 1.5648332

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6504239, upper bound: 0.6600853
time: 3.73 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6556010, upper bound: 0.6623011
time: 5.67 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -13.3069401, -10.4902096, -13.1606417, -10.4756079, -1.7391410, 1.6290171
1: -11.4256086, -8.4348717, -11.2988663, -8.3863068, -2.0927410, 1.9218879
2: -10.7750120, -8.5727234, -10.7254848, -8.5369482, -1.8425822, 1.6738737
3: -4.5699735, -2.3410995, -4.4253130, -2.2959745, -1.5904231, 1.5868900
4: -15.1519194, -12.4279480, -15.1697912, -12.5018415, -1.7609096, 1.6214530
5: 8.2221375, 9.6680603, 8.2235432, 9.7037125, -1.0884252, 1.0219696
6: -4.7887893, -2.3123035, -4.7399817, -2.2900791, -1.5849962, 1.6299329
7: -15.7892513, -12.9876451, -15.7442493, -12.9353447, -2.2491293, 1.8552761
8: -0.7940555, 0.9715207, -0.8271551, 0.9180839, -1.0310955, 1.3003058
9: -6.6794038, -4.9594150, -6.6974626, -5.0055976, -1.2154655, 1.5892963

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6571213, upper bound: 0.6600852
time: 4.09 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623010, upper bound: 0.6623016
time: 5.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.60 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6626736, upper bound: 0.6760375
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6760357
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6693374
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6693736, upper bound: 0.6760373
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6504239, upper bound: 0.6600853
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6556010, upper bound: 0.6623011
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6571213, upper bound: 0.6600852
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 15.60
Output dim: 5, lower bound: -0.6623010, upper bound: 0.6623016

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1633673, -10.5471478, -13.1613684, -10.5349464, -1.5889769, 1.5805178
1: -11.3239450, -8.4196949, -11.2994976, -8.4160891, -1.9023070, 1.8825910
2: -10.7499352, -8.5666199, -10.7255192, -8.5550365, -1.7006361, 1.6925733
3: -4.3989763, -2.3425825, -4.4147496, -2.3468454, -1.5426612, 1.5437863
4: -15.1122828, -12.5838108, -15.1385670, -12.5457249, -1.7049098, 1.6874962
5: 8.2515154, 9.7071228, 8.2406349, 9.7064877, -1.0012093, 1.0153364
6: -4.7121363, -2.3152876, -4.7284250, -2.3063426, -1.5819407, 1.5855207
7: -15.7508335, -12.9453793, -15.7444878, -12.9394283, -2.1308899, 2.1133261
8: -0.7826340, 0.9031253, -0.8066788, 0.9184277, -1.0650530, 1.0732439
9: -6.6767054, -5.0664907, -6.6816726, -5.0292954, -1.5163612, 1.4837480

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6685886, upper bound: 0.6781950
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6730911, upper bound: 0.6797284
time: 4.37 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1613560, -10.5350914, -13.1613798, -10.5291605, -1.5990219, 1.5806661
1: -11.2994614, -8.4265642, -11.2995005, -8.4167728, -1.8843102, 1.8918495
2: -10.7255049, -8.5611706, -10.7255211, -8.5464029, -1.7428761, 1.6718154
3: -4.4181614, -2.3468671, -4.4255018, -2.3468423, -1.5341806, 1.5768094
4: -15.1385651, -12.5070190, -15.1385698, -12.5021210, -1.7590604, 1.6859665
5: 8.2271509, 9.7064800, 8.2232246, 9.7064896, -1.0078900, 1.0350884
6: -4.7387886, -2.3063402, -4.7410054, -2.3063388, -1.5748610, 1.6143417
7: -15.7444801, -12.9415779, -15.7444859, -12.9367161, -2.1237144, 2.1291656
8: -0.8204088, 0.9184208, -0.8223681, 0.9184284, -1.0567217, 1.1088593
9: -6.6816416, -5.0141444, -6.6816797, -5.0055199, -1.5242124, 1.5082068

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 158

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6752802, upper bound: 0.6781949
time: 3.98 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6797262, upper bound: 0.6797284
time: 4.11 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.1613684, -10.5349464, -13.3090773, -10.5022688, -1.6078849, 1.6656575
1: -11.2994976, -8.4160891, -11.4500942, -8.4280024, -1.9058838, 2.0809333
2: -10.7255192, -8.5550365, -10.7994423, -8.5782375, -1.6968141, 1.7827580
3: -4.4147496, -2.3468454, -4.5506849, -2.3368161, -1.5417118, 1.5515802
4: -15.1385670, -12.5457249, -15.1256447, -12.5048466, -1.5901875, 1.6974134
5: 8.2406349, 9.7064877, 8.2465706, 9.6687326, -1.0172110, 1.0576403
6: -4.7284250, -2.3063426, -4.7621164, -2.3212516, -1.5826411, 1.5623834
7: -15.7444878, -12.9394283, -15.7956581, -12.9913330, -1.8326612, 2.2255487
8: -0.8066788, 0.9184277, -0.7562866, 0.9562299, -1.2069826, 1.0684273
9: -6.6816726, -5.0292954, -6.6745906, -5.0117588, -1.5415492, 1.1277103

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570878
time: 4.30 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622648
time: 3.62 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.1613798, -10.5291605, -13.3069401, -10.4902096, -1.6080317, 1.6874726
1: -11.2995005, -8.4167728, -11.4256086, -8.4348717, -1.9151430, 2.0629249
2: -10.7255211, -8.5464029, -10.7750120, -8.5727234, -1.6761885, 1.8249974
3: -4.4255018, -2.3468423, -4.5699735, -2.3410995, -1.5746841, 1.5496128
4: -15.1385698, -12.5021210, -15.1519194, -12.4279480, -1.5997343, 1.7515416
5: 8.2232246, 9.7064896, 8.2221375, 9.6680603, -1.0423954, 1.0643258
6: -4.7410054, -2.3063388, -4.7887893, -2.3123035, -1.6114659, 1.5590606
7: -15.7444859, -12.9367161, -15.7892513, -12.9876451, -1.8541832, 2.2183080
8: -0.8223681, 0.9184284, -0.7940555, 0.9715207, -1.2425976, 1.0632164
9: -6.6816797, -5.0055199, -6.6794038, -4.9594150, -1.5660076, 1.1893837

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 158

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637878
time: 4.03 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689648
time: 4.29 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3071823, -10.5114536, -13.1558962, -10.5026989, -1.6808958, 1.6099105
1: -11.4477100, -8.4399529, -11.2928782, -8.4176598, -2.0532374, 1.8853056
2: -10.7732630, -8.5800972, -10.6638069, -8.5507116, -1.7706437, 1.6202850
3: -4.5499749, -2.3376729, -4.4127197, -2.2981317, -1.5897160, 1.5520051
4: -15.1255608, -12.5076370, -15.1695929, -12.5530024, -1.6998305, 1.6054232
5: 8.2469168, 9.6606894, 8.2418222, 9.6817417, -1.0586271, 0.9817870
6: -4.7614303, -2.3335683, -4.7253838, -2.3213990, -1.5464053, 1.5839441
7: -15.7939177, -12.9961119, -15.7399340, -12.9510145, -2.2431393, 1.8274395
8: -0.7472873, 0.9561176, -0.7871320, 0.9178238, -1.0219457, 1.2379141
9: -6.6714001, -5.0126677, -6.6887412, -5.0317678, -1.1466279, 1.5536976

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6504239, upper bound: 0.6600853
time: 4.30 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6504239, upper bound: 0.6600852
time: 4.38 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -13.3082132, -10.5061989, -13.1856918, -10.4909334, -1.6900520, 1.6787150
1: -11.4490948, -8.4504070, -11.3026133, -8.4329481, -2.0510139, 1.9773092
2: -10.7872429, -8.5788212, -10.7027397, -8.4931984, -1.8839674, 1.6377728
3: -4.5504627, -2.3372984, -4.4189887, -2.2946861, -1.5931363, 1.5589857
4: -15.1256056, -12.5112572, -15.1663752, -12.5607138, -1.7033997, 1.6098258
5: 8.2467575, 9.6611557, 8.2292767, 9.6859436, -1.0701780, 0.9981060
6: -4.7617340, -2.3275123, -4.7556210, -2.2980926, -1.5585141, 1.6469655
7: -15.7951097, -12.9967470, -15.7500772, -12.9492884, -2.2432575, 1.8377299
8: -0.7527628, 0.9561768, -0.8060360, 0.9442966, -1.0519500, 1.2612667
9: -6.6699343, -5.0121303, -6.6871572, -5.0285902, -1.1730235, 1.5544095

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6556010, upper bound: 0.6623010
time: 9.62 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6556010, upper bound: 0.6623013
time: 4.91 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -13.3050594, -10.4993925, -13.1559067, -10.4969168, -1.7026877, 1.6099453
1: -11.4232244, -8.4468231, -11.2928829, -8.4183407, -2.0352206, 1.8945911
2: -10.7488317, -8.5745049, -10.6638069, -8.5418549, -1.8129005, 1.5996656
3: -4.5692968, -2.3419483, -4.4234729, -2.2981308, -1.5877519, 1.5849907
4: -15.1518402, -12.4307404, -15.1695967, -12.5094280, -1.7539644, 1.6149971
5: 8.2224941, 9.6599913, 8.2244072, 9.6817436, -1.0653749, 1.0069976
6: -4.7880993, -2.3246164, -4.7379608, -2.3213968, -1.5432248, 1.6127725
7: -15.7874365, -12.9924278, -15.7399292, -12.9483185, -2.2358065, 1.8490682
8: -0.7850575, 0.9714088, -0.8028193, 0.9178240, -1.0167532, 1.2735286
9: -6.6762142, -4.9603271, -6.6887474, -5.0079937, -1.2082412, 1.5781603

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6571213, upper bound: 0.6600853
time: 4.50 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6571213, upper bound: 0.6600853
time: 4.00 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -13.3060932, -10.4941368, -13.1857033, -10.4851532, -1.7118678, 1.6787047
1: -11.4246092, -8.4572773, -11.3026152, -8.4336290, -2.0330057, 1.9865606
2: -10.7628136, -8.5732822, -10.7027416, -8.4844170, -1.9262190, 1.6171601
3: -4.5697613, -2.3415761, -4.4297423, -2.2946842, -1.5911703, 1.5919700
4: -15.1518841, -12.4343634, -15.1663771, -12.5171385, -1.7575302, 1.6194165
5: 8.2223282, 9.6604738, 8.2118702, 9.6859455, -1.0769055, 1.0233948
6: -4.7883987, -2.3185630, -4.7681975, -2.2980893, -1.5554841, 1.6757908
7: -15.7886810, -12.9930592, -15.7500744, -12.9465990, -2.2359886, 1.8593216
8: -0.7905314, 0.9714684, -0.8217249, 0.9442992, -1.0467691, 1.2968812
9: -6.6747465, -4.9597888, -6.6871638, -5.0048151, -1.2346423, 1.5788717

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2375
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 2375

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623010, upper bound: 0.6623012
time: 5.72 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6623010, upper bound: 0.6623018
time: 5.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.86 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6685886, upper bound: 0.6781950
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6730911, upper bound: 0.6797284
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6752802, upper bound: 0.6781949
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6797262, upper bound: 0.6797284
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6570878
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6622648
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6600832, upper bound: 0.6637878
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6623012, upper bound: 0.6689648
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6504239, upper bound: 0.6600853
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6504239, upper bound: 0.6600852
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6556010, upper bound: 0.6623010
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6556010, upper bound: 0.6623013
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6571213, upper bound: 0.6600853
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6571213, upper bound: 0.6600853
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6623010, upper bound: 0.6623012
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.86
Output dim: 5, lower bound: -0.6623010, upper bound: 0.6623018

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1615391, -10.5563307, -13.1566467, -10.5562611, -1.5485258, 1.5615640
1: -11.3216839, -8.4316435, -11.2935266, -8.4481239, -1.8450642, 1.8552706
2: -10.7237625, -8.5686007, -10.6638412, -8.5604353, -1.6708791, 1.6182928
3: -4.3982882, -2.3434389, -4.4129162, -2.3489814, -1.5400195, 1.5418971
4: -15.1122036, -12.5866070, -15.1383734, -12.5532913, -1.6979523, 1.6822534
5: 8.2518520, 9.6990671, 8.2414856, 9.6845083, -0.9779398, 1.0012695
6: -4.7113757, -2.3275948, -4.7264080, -2.3376534, -1.5394897, 1.5683572
7: -15.7493277, -12.9501801, -15.7401829, -12.9524250, -2.1179848, 2.1050787
8: -0.7736735, 0.9030194, -0.7824209, 0.9181695, -1.0506201, 1.0469103
9: -6.6735115, -5.0673761, -6.6729603, -5.0316868, -1.5096364, 1.4726868

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6649649, upper bound: 0.6635229
time: 4.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6652419, upper bound: 0.6748530
time: 4.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1625299, -10.5510769, -13.1863976, -10.5444908, -1.5598545, 1.6303287
1: -11.3229933, -8.4421015, -11.3032370, -8.4634132, -1.8426452, 1.9471924
2: -10.7377396, -8.5672417, -10.7027769, -8.5028753, -1.7841959, 1.6356843
3: -4.3987608, -2.3430641, -4.4191785, -2.3455374, -1.5430021, 1.5488718
4: -15.1122456, -12.5902300, -15.1351528, -12.5610037, -1.7015419, 1.6754336
5: 8.2516966, 9.6995344, 8.2289534, 9.6887178, -0.9895391, 1.0119468
6: -4.7117066, -2.3215444, -4.7565508, -2.3143501, -1.5495863, 1.6313269
7: -15.7503586, -12.9508038, -15.7503300, -12.9507360, -2.1177721, 2.1074896
8: -0.7791660, 0.9030747, -0.8013029, 0.9446208, -1.0753777, 1.0705634
9: -6.6720467, -5.0668530, -6.6713777, -5.0285101, -1.5217514, 1.4733491

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6696085, upper bound: 0.6650625
time: 4.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6697274, upper bound: 0.6763650
time: 4.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1595364, -10.5442724, -13.1566563, -10.5504742, -1.5586152, 1.5615993
1: -11.2972002, -8.4385118, -11.2935266, -8.4488049, -1.8270588, 1.8645539
2: -10.6993294, -8.5630732, -10.6638441, -8.5515947, -1.7131329, 1.5975413
3: -4.4175091, -2.3477132, -4.4236708, -2.3489780, -1.5315223, 1.5749271
4: -15.1384869, -12.5098171, -15.1383743, -12.5097094, -1.7521067, 1.6807151
5: 8.2275000, 9.6984005, 8.2240858, 9.6845102, -0.9846790, 1.0209950
6: -4.7380171, -2.3186455, -4.7389874, -2.3376493, -1.5323980, 1.5971832
7: -15.7429028, -12.9463787, -15.7401829, -12.9497242, -2.1107168, 2.1211677
8: -0.8114500, 0.9183147, -0.7981102, 0.9181702, -1.0422890, 1.0825258
9: -6.6784458, -5.0150347, -6.6729646, -5.0079112, -1.5174737, 1.4971447

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6716564, upper bound: 0.6635228
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6719337, upper bound: 0.6748530
time: 4.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1605358, -10.5390167, -13.1864090, -10.5387077, -1.5699253, 1.6303146
1: -11.2985115, -8.4489708, -11.3032408, -8.4640923, -1.8246498, 1.9564419
2: -10.7133093, -8.5617695, -10.7027817, -8.4941101, -1.8264451, 1.6149387
3: -4.4179573, -2.3473399, -4.4299331, -2.3455355, -1.5344934, 1.5818992
4: -15.1385279, -12.5134401, -15.1351528, -12.5174198, -1.7556953, 1.6738837
5: 8.2273388, 9.6988859, 8.2115650, 9.6887207, -0.9962573, 1.0316813
6: -4.7383471, -2.3125937, -4.7691312, -2.3143466, -1.5424821, 1.6601505
7: -15.7439833, -12.9470024, -15.7503271, -12.9480457, -2.1105700, 2.1234884
8: -0.8169415, 0.9183688, -0.8169909, 0.9446225, -1.0670466, 1.1061798
9: -6.6769814, -5.0145102, -6.6713853, -5.0047336, -1.5295792, 1.4978085

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6762606, upper bound: 0.6650626
time: 4.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6763629, upper bound: 0.6763650
time: 4.32 seconds

## BFS IS instance: IS_A1_B2_B1_A1

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

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6472590, upper bound: 0.6536343
time: 3.71 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6554495, upper bound: 0.6536785
time: 4.36 seconds

## BFS IS instance: IS_A1_B2_B1_A2

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

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6494775, upper bound: 0.6588114
time: 3.83 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6576674, upper bound: 0.6588557
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_B2_A1

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

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6472590, upper bound: 0.6603324
time: 6.21 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6554495, upper bound: 0.6603786
time: 4.18 seconds

## BFS IS instance: IS_A1_B2_B2_A2

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

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1836

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6494775, upper bound: 0.6655113
time: 4.23 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6576674, upper bound: 0.6655532
time: 4.38 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -13.3071823, -10.5114536, -13.1566467, -10.5562611, -1.6292205, 1.5889311
1: -11.4477100, -8.4399529, -11.2935266, -8.4481239, -2.0234208, 1.8785639
2: -10.7732630, -8.5800972, -10.6638412, -8.5604353, -1.7530086, 1.6224113
3: -4.5499749, -2.3376729, -4.4129162, -2.3489814, -1.5491314, 1.5397177
4: -15.1255608, -12.5076370, -15.1383734, -12.5532913, -1.6904845, 1.5836763
5: 8.2469168, 9.6606894, 8.2414856, 9.6845083, -1.0343513, 1.0023434
6: -4.7614303, -2.3335683, -4.7264080, -2.3376534, -1.5204983, 1.5654757
7: -15.7939177, -12.9961119, -15.7401829, -12.9524250, -2.2122507, 1.8263929
8: -0.7472873, 0.9561176, -0.7824209, 0.9181695, -1.0538712, 1.1806483
9: -6.6714001, -5.0126677, -6.6729603, -5.0316868, -1.1205461, 1.5304098

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6457895, upper bound: 0.6472606
time: 4.70 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6457903, upper bound: 0.6554515
time: 4.44 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -13.3071823, -10.5114536, -13.3020611, -10.5113811, -1.5290985, 1.5450342
1: -11.4477100, -8.4399529, -11.4193535, -8.4564342, -1.8730516, 1.8832250
2: -10.7732630, -8.5800972, -10.7133245, -8.5717316, -1.6679335, 1.6155338
3: -4.5499749, -2.3376729, -4.5645933, -2.3432310, -1.3436995, 1.3501594
4: -15.1255608, -12.5076370, -15.1517258, -12.4743414, -1.5184679, 1.5048151
5: 8.2469168, 9.6606894, 8.2365532, 9.6461134, -0.9505900, 0.9722964
6: -4.7614303, -2.3335683, -4.7766171, -2.3436375, -1.4350538, 1.4634264
7: -15.7939177, -12.9961119, -15.7843122, -12.9983559, -1.8391685, 1.8292568
8: -0.7472873, 0.9561176, -0.7559652, 0.9712560, -1.0188832, 1.0124434
9: -6.6714001, -5.0126677, -6.6707249, -4.9770184, -1.1465583, 1.1300542

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6457895, upper bound: 0.6472611
time: 4.38 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6457903, upper bound: 0.6554495
time: 4.44 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -13.3082132, -10.5061989, -13.1863976, -10.5444908, -1.6383886, 1.6576965
1: -11.4490948, -8.4504070, -11.3032370, -8.4634132, -2.0211978, 1.9704843
2: -10.7872429, -8.5788212, -10.7027769, -8.5028753, -1.8663216, 1.6399970
3: -4.5504627, -2.3372984, -4.4191785, -2.3455374, -1.5526204, 1.5467162
4: -15.1256056, -12.5112572, -15.1351528, -12.5610037, -1.6940465, 1.5879560
5: 8.2467575, 9.6611557, 8.2289534, 9.6887178, -1.0459628, 1.0185097
6: -4.7617340, -2.3275123, -4.7565508, -2.3143501, -1.5326223, 1.6284456
7: -15.7951097, -12.9967470, -15.7503300, -12.9507360, -2.2123070, 1.8366587
8: -0.7527628, 0.9561768, -0.8013029, 0.9446208, -1.0839009, 1.2042990
9: -6.6699343, -5.0121303, -6.6713777, -5.0285101, -1.1469378, 1.5311251

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6509666, upper bound: 0.6494796
time: 4.04 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6509673, upper bound: 0.6576674
time: 5.87 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -13.3082132, -10.5061989, -13.3320122, -10.4996128, -1.5381875, 1.6091356
1: -11.4490948, -8.4504070, -11.4292049, -8.4717159, -1.8706193, 1.9755235
2: -10.7872429, -8.5788212, -10.7522335, -8.5140104, -1.7813287, 1.6330183
3: -4.5504627, -2.3372984, -4.5708861, -2.3397861, -1.3471293, 1.3557200
4: -15.1256056, -12.5112572, -15.1485224, -12.4820595, -1.5215473, 1.5094020
5: 8.2467575, 9.6611557, 8.2239780, 9.6503048, -0.9618441, 0.9894140
6: -4.7617340, -2.3275123, -4.8073292, -2.3203678, -1.4472008, 1.5226295
7: -15.7951097, -12.9967470, -15.7946587, -12.9966345, -1.8436470, 1.8395667
8: -0.7527628, 0.9561768, -0.7747347, 0.9977977, -1.0489919, 1.0350249
9: -6.6699343, -5.0121303, -6.6690583, -4.9738240, -1.1729672, 1.1339469

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6509666, upper bound: 0.6494776
time: 4.11 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6509673, upper bound: 0.6576695
time: 5.03 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.3050594, -10.4993925, -13.1566563, -10.5504742, -1.6510093, 1.5889657
1: -11.4232244, -8.4468231, -11.2935266, -8.4488049, -2.0054040, 1.8878474
2: -10.7488317, -8.5745049, -10.6638441, -8.5515947, -1.7952609, 1.6017916
3: -4.5692968, -2.3419483, -4.4236708, -2.3489780, -1.5471513, 1.5726979
4: -15.1518402, -12.4307404, -15.1383743, -12.5097094, -1.7446170, 1.5932360
5: 8.2224941, 9.6599913, 8.2240858, 9.6845102, -1.0410953, 1.0274993
6: -4.7880993, -2.3246164, -4.7389874, -2.3376493, -1.5172930, 1.5943041
7: -15.7874365, -12.9924278, -15.7401829, -12.9497242, -2.2049179, 1.8480127
8: -0.7850575, 0.9714088, -0.7981102, 0.9181702, -1.0486672, 1.2162631
9: -6.6762142, -4.9603271, -6.6729646, -5.0079112, -1.1821640, 1.5548677

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6524895, upper bound: 0.6472611
time: 4.30 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6524902, upper bound: 0.6554516
time: 3.99 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.3050594, -10.4993925, -13.3020706, -10.5055943, -1.5508840, 1.5362804
1: -11.4232244, -8.4468231, -11.4193573, -8.4571171, -1.8550305, 1.8925292
2: -10.7488317, -8.5745049, -10.7133274, -8.5628185, -1.7096133, 1.5949159
3: -4.5692968, -2.3419483, -4.5753388, -2.3432283, -1.3417914, 1.3752332
4: -15.1518402, -12.4307404, -15.1517277, -12.4306383, -1.5681748, 1.5144353
5: 8.2224941, 9.6599913, 8.2190933, 9.6461124, -0.9535214, 0.9975153
6: -4.7880993, -2.3246164, -4.7891903, -2.3436341, -1.4318779, 1.4878910
7: -15.7874365, -12.9924278, -15.7843113, -12.9957619, -1.8275514, 1.8509064
8: -0.7850575, 0.9714088, -0.7716520, 0.9712572, -1.0137033, 1.0456065
9: -6.6762142, -4.9603271, -6.6707234, -4.9532433, -1.2081642, 1.1110904

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6524895, upper bound: 0.6472611
time: 5.50 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6524902, upper bound: 0.6554497
time: 4.83 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -13.3060932, -10.4941368, -13.1864090, -10.5387077, -1.6601996, 1.6576815
1: -11.4246092, -8.4572773, -11.3032408, -8.4640923, -2.0031891, 1.9797344
2: -10.7628136, -8.5732822, -10.7027817, -8.4941101, -1.9085698, 1.6193852
3: -4.5697613, -2.3415761, -4.4299331, -2.3455355, -1.5506375, 1.5796938
4: -15.1518841, -12.4343634, -15.1351528, -12.5174198, -1.7481771, 1.5975323
5: 8.2223282, 9.6604738, 8.2115650, 9.6887207, -1.0526855, 1.0437448
6: -4.7883987, -2.3185630, -4.7691312, -2.3143466, -1.5295680, 1.6572723
7: -15.7886810, -12.9930592, -15.7503271, -12.9480457, -2.2050381, 1.8582435
8: -0.7905314, 0.9714684, -0.8169909, 0.9446225, -1.0787098, 1.2399137
9: -6.6747465, -4.9597888, -6.6713853, -5.0047336, -1.2085605, 1.5555830

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6576666, upper bound: 0.6494795
time: 4.83 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6576673, upper bound: 0.6576695
time: 3.85 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -13.3060932, -10.4941368, -13.3320208, -10.4938269, -1.5599961, 1.6004200
1: -11.4246092, -8.4572773, -11.4292088, -8.4723969, -1.8526077, 1.9847922
2: -10.7628136, -8.5732822, -10.7522354, -8.5051794, -1.8230019, 1.6124058
3: -4.5697613, -2.3415761, -4.5816317, -2.3397834, -1.3452182, 1.3807933
4: -15.1518841, -12.4343634, -15.1485252, -12.4383535, -1.5712600, 1.5190392
5: 8.2223282, 9.6604738, 8.2065287, 9.6503048, -0.9647244, 1.0147117
6: -4.7883987, -2.3185630, -4.8199048, -2.3203635, -1.4441748, 1.5471087
7: -15.7886810, -12.9930592, -15.7946606, -12.9940453, -1.8320370, 1.8611774
8: -0.7905314, 0.9714684, -0.7904239, 0.9978006, -1.0438235, 1.0681885
9: -6.6747465, -4.9597888, -6.6690540, -4.9500480, -1.2345805, 1.1149819

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1836

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6576666, upper bound: 0.6494795
time: 4.01 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6576673, upper bound: 0.6576695
time: 3.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.63 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6649649, upper bound: 0.6635229
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6652419, upper bound: 0.6748530
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6696085, upper bound: 0.6650625
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6697274, upper bound: 0.6763650
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6716564, upper bound: 0.6635228
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6719337, upper bound: 0.6748530
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6762606, upper bound: 0.6650626
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6763629, upper bound: 0.6763650
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6472590, upper bound: 0.6536343
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6554495, upper bound: 0.6536785
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6494775, upper bound: 0.6588114
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6576674, upper bound: 0.6588557
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6472590, upper bound: 0.6603324
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6554495, upper bound: 0.6603786
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6494775, upper bound: 0.6655113
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6576674, upper bound: 0.6655532
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6457895, upper bound: 0.6472606
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6457903, upper bound: 0.6554515
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6457895, upper bound: 0.6472611
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6457903, upper bound: 0.6554495
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6509666, upper bound: 0.6494796
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6509673, upper bound: 0.6576674
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6509666, upper bound: 0.6494776
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6509673, upper bound: 0.6576695
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6524895, upper bound: 0.6472611
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6524902, upper bound: 0.6554516
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6524895, upper bound: 0.6472611
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6524902, upper bound: 0.6554497
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6576666, upper bound: 0.6494795
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6576673, upper bound: 0.6576695
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6576666, upper bound: 0.6494795
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.63
Output dim: 5, lower bound: -0.6576673, upper bound: 0.6576695

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -13.0900173, -10.5959167, -13.1306915, -10.5562706, -1.4915004, 1.4866495
1: -11.3219852, -8.4524155, -11.2909002, -8.4553823, -1.8084774, 1.8380580
2: -10.7186346, -8.5827971, -10.6618223, -8.5642605, -1.6489286, 1.5834353
3: -4.3991318, -2.3512614, -4.4118752, -2.3519464, -1.5388565, 1.5327404
4: -15.1069555, -12.5758057, -15.1365376, -12.5537252, -1.6859298, 1.6738839
5: 8.2622089, 9.6767502, 8.2415562, 9.6765289, -0.9641436, 0.9810481
6: -4.6440992, -2.3782482, -4.7026982, -2.3376732, -1.4826322, 1.4901471
7: -15.7326221, -12.9745512, -15.7377262, -12.9609308, -2.0851688, 2.0748839
8: -0.7717934, 0.8993661, -0.7817214, 0.9168925, -1.0457053, 1.0421069
9: -6.6699696, -5.0690551, -6.6717434, -5.0328526, -1.5010223, 1.4648490

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6551738, upper bound: 0.6545893
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6571692, upper bound: 0.6545874
time: 5.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.1412029, -10.5563869, -13.1495180, -10.5562754, -1.4828172, 1.5439813
1: -11.3180056, -8.4549217, -11.2922573, -8.4557257, -1.8343706, 1.8101423
2: -10.7194185, -8.5735941, -10.6623688, -8.5620155, -1.6405053, 1.6122029
3: -4.3981233, -2.3588123, -4.4128523, -2.3542085, -1.5346689, 1.5295548
4: -15.0943861, -12.5871449, -15.1325455, -12.5534697, -1.6772075, 1.6776943
5: 8.2523012, 9.6931963, 8.2416325, 9.6824579, -0.9680252, 0.9934815
6: -4.7007217, -2.3276656, -4.7219896, -2.3376770, -1.4500721, 1.5622246
7: -15.7488947, -12.9638786, -15.7400103, -12.9569006, -2.1143360, 2.0964017
8: -0.7715952, 0.8997872, -0.7816978, 0.9171162, -1.0461819, 1.0418720
9: -6.6669941, -5.0687761, -6.6708274, -5.0321670, -1.5022311, 1.4687581

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6574457, upper bound: 0.6691948
time: 3.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6574457, upper bound: 0.6659317
time: 4.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.0909100, -10.5906639, -13.1605310, -10.5445061, -1.5027165, 1.5554943
1: -11.3233004, -8.4628716, -11.3006821, -8.4706736, -1.8061247, 1.9301047
2: -10.7326279, -8.5814943, -10.7007542, -8.5064163, -1.7622552, 1.6009088
3: -4.3996148, -2.3508680, -4.4181957, -2.3484979, -1.5418859, 1.5397518
4: -15.1070004, -12.5794258, -15.1333160, -12.5614262, -1.6895485, 1.6670861
5: 8.2620487, 9.6771917, 8.2290249, 9.6807022, -0.9757478, 0.9916905
6: -4.6444225, -2.3721910, -4.7328625, -2.3143692, -1.4926944, 1.5531559
7: -15.7336788, -12.9751740, -15.7478466, -12.9592333, -2.0850792, 2.0771852
8: -0.7772534, 0.8994222, -0.8005610, 0.9433446, -1.0704105, 1.0657915
9: -6.6685038, -5.0685387, -6.6701612, -5.0296564, -1.5132055, 1.4655128

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6595617, upper bound: 0.6560062
time: 4.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6616119, upper bound: 0.6560061
time: 4.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -13.1421776, -10.5511332, -13.1792736, -10.5445099, -1.4939582, 1.6127584
1: -11.3193159, -8.4653788, -11.3019638, -8.4710150, -1.8319488, 1.9020736
2: -10.7334108, -8.5723419, -10.7013254, -8.5044823, -1.7538743, 1.6294866
3: -4.3985977, -2.3584073, -4.4191179, -2.3507521, -1.5376687, 1.5365901
4: -15.0944271, -12.5907736, -15.1293201, -12.5611839, -1.6807866, 1.6708689
5: 8.2521477, 9.6936569, 8.2291021, 9.6866627, -0.9796343, 1.0040609
6: -4.7010498, -2.3216133, -4.7521086, -2.3143733, -1.4600432, 1.6251950
7: -15.7499304, -12.9645004, -15.7501545, -12.9552126, -2.1141262, 2.0988073
8: -0.7771122, 0.8998420, -0.8006105, 0.9435668, -1.0708663, 1.0654755
9: -6.6655302, -5.0682530, -6.6692448, -5.0289845, -1.5143461, 1.4694147

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6617135, upper bound: 0.6706271
time: 3.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6617135, upper bound: 0.6673088
time: 4.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -13.0878487, -10.5838575, -13.1307049, -10.5504866, -1.5015535, 1.4866822
1: -11.2975016, -8.4592848, -11.2909031, -8.4560642, -1.7904038, 1.8473735
2: -10.6942101, -8.5770512, -10.6618204, -8.5553007, -1.6911969, 1.5626850
3: -4.4184575, -2.3555651, -4.4226303, -2.3519435, -1.5303602, 1.5657530
4: -15.1332426, -12.4989748, -15.1365433, -12.5101528, -1.7400832, 1.6723449
5: 8.2378492, 9.6760149, 8.2241573, 9.6765308, -0.9709177, 1.0007104
6: -4.6708164, -2.3692987, -4.7152753, -2.3376703, -1.4755464, 1.5189729
7: -15.7259827, -12.9707537, -15.7377262, -12.9582272, -2.0776377, 2.0911155
8: -0.8095679, 0.9146636, -0.7974107, 0.9168944, -1.0373745, 1.0777221
9: -6.6748972, -5.0167103, -6.6717510, -5.0090761, -1.5088620, 1.4893093

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6716563, upper bound: 0.6542133
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6716565, upper bound: 0.6542133
time: 5.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -13.1391115, -10.5443296, -13.1495314, -10.5504913, -1.4928331, 1.5440431
1: -11.2935228, -8.4617901, -11.2922621, -8.4564047, -1.8163342, 1.8194373
2: -10.6949949, -8.5680504, -10.6623688, -8.5531683, -1.6827660, 1.5914495
3: -4.4173517, -2.3631260, -4.4236050, -2.3542037, -1.5261784, 1.5625603
4: -15.1206694, -12.5103264, -15.1325474, -12.5098839, -1.7313595, 1.6761563
5: 8.2279348, 9.6925230, 8.2242327, 9.6824598, -0.9747664, 1.0131997
6: -4.7274361, -2.3187175, -4.7345695, -2.3376741, -1.4429862, 1.5910501
7: -15.7424507, -12.9600773, -15.7400074, -12.9541979, -2.1070442, 2.1125016
8: -0.8093703, 0.9150858, -0.7973878, 0.9171162, -1.0378516, 1.0774870
9: -6.6719246, -5.0164247, -6.6708355, -5.0083919, -1.5100746, 1.4932184

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6719335, upper bound: 0.6655319
time: 3.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6719337, upper bound: 0.6655318
time: 3.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.0887489, -10.5786057, -13.1605415, -10.5387220, -1.5127506, 1.5554767
1: -11.2988148, -8.4697390, -11.3006859, -8.4713554, -1.7880616, 1.9393837
2: -10.7082014, -8.5758076, -10.7007551, -8.4975376, -1.8045163, 1.5801635
3: -4.4189124, -2.3551707, -4.4289474, -2.3484936, -1.5333796, 1.5727623
4: -15.1332836, -12.5025959, -15.1333189, -12.5178509, -1.7437000, 1.6655347
5: 8.2376823, 9.6764727, 8.2116356, 9.6807041, -0.9824996, 1.0113633
6: -4.6711397, -2.3632424, -4.7454391, -2.3143673, -1.4855945, 1.5819798
7: -15.7270937, -12.9713764, -15.7478437, -12.9565420, -2.0776186, 2.0933199
8: -0.8150282, 0.9147191, -0.8162510, 0.9433463, -1.0620790, 1.1014069
9: -6.6734328, -5.0161924, -6.6701684, -5.0058818, -1.5210328, 1.4899726

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6762605, upper bound: 0.6558192
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6762607, upper bound: 0.6558191
time: 3.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -13.1400928, -10.5390768, -13.1792870, -10.5387211, -1.5039554, 1.6127701
1: -11.2948313, -8.4722462, -11.3019676, -8.4716959, -1.8139210, 1.9113352
2: -10.7089863, -8.5668526, -10.7013254, -8.4957085, -1.7961297, 1.6087389
3: -4.4178019, -2.3627234, -4.4298735, -2.3507490, -1.5291686, 1.5695930
4: -15.1207085, -12.5139561, -15.1293278, -12.5175972, -1.7349377, 1.6693180
5: 8.2277756, 9.6930008, 8.2117119, 9.6866646, -0.9863559, 1.0237886
6: -4.7277608, -2.3126647, -4.7646885, -2.3143692, -1.4529438, 1.6540179
7: -15.7435389, -12.9606981, -15.7501535, -12.9525166, -2.1068993, 2.1148167
8: -0.8148880, 0.9151423, -0.8162985, 0.9435701, -1.0625355, 1.1010909
9: -6.6704617, -5.0159006, -6.6692529, -5.0052099, -1.5221772, 1.4938750

Time for backsubstitution: 5.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 3127
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 677

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6763628, upper bound: 0.6671214
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6763629, upper bound: 0.6671215
time: 3.88 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.1306915, -10.5562706, -13.2361183, -10.5454826, -1.5180306, 1.5654745
1: -11.2909002, -8.4553823, -11.4503298, -8.4527369, -1.8660345, 2.0128927
2: -10.6618223, -8.5642605, -10.7676086, -8.5927334, -1.5892272, 1.7341514
3: -4.4118752, -2.3519464, -4.5554857, -2.3427744, -1.5333500, 1.5500662
4: -15.1365376, -12.5537252, -15.1203175, -12.4951077, -1.5774570, 1.6828046
5: 8.2415562, 9.6765289, 8.2546854, 9.6394339, -0.9846144, 1.0227261
6: -4.7026982, -2.3376732, -4.7037630, -2.3794785, -1.4989982, 1.4694149
7: -15.7377262, -12.9609308, -15.7882118, -13.0125408, -1.8079138, 2.1953049
8: -0.7817214, 0.9168925, -0.7455928, 0.9524357, -1.1752696, 1.0497086
9: -6.6717434, -5.0328526, -6.6683726, -5.0137129, -1.5232081, 1.1170022

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6417365, upper bound: 0.6458986
time: 4.64 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6384311, upper bound: 0.6458987
time: 3.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -13.1495180, -10.5562754, -13.2830086, -10.5115061, -1.5730076, 1.5561192
1: -11.2922573, -8.4557257, -11.4440680, -8.4704237, -1.8348927, 2.0124974
2: -10.6623688, -8.5620155, -10.7694101, -8.5855942, -1.6127729, 1.7268033
3: -4.4128523, -2.3542085, -4.5498047, -2.3563483, -1.5283175, 1.5432644
4: -15.1325455, -12.5534697, -15.1077232, -12.5081692, -1.5810423, 1.6737432
5: 8.2416325, 9.6824579, 8.2473621, 9.6534691, -0.9981170, 1.0247214
6: -4.7219896, -2.3376770, -4.7488832, -2.3336358, -1.5593400, 1.4306502
7: -15.7400103, -12.9569006, -15.7934208, -13.0168991, -1.8038993, 2.2084966
8: -0.7816978, 0.9171162, -0.7453618, 0.9529214, -1.1757712, 1.0506518
9: -6.6708274, -5.0321670, -6.6645164, -5.0140653, -1.5262632, 1.1128924

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6499264, upper bound: 0.6459411
time: 5.47 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6466211, upper bound: 0.6459430
time: 4.19 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.1605310, -10.5445061, -13.2370558, -10.5402260, -1.5868754, 1.5745564
1: -11.3006821, -8.4706736, -11.4517336, -8.4631891, -1.9580812, 2.0106390
2: -10.7007542, -8.5064163, -10.7816067, -8.5915012, -1.6068459, 1.8474755
3: -4.4181957, -2.3484979, -4.5559764, -2.3423796, -1.5403824, 1.5535579
4: -15.1333160, -12.5614262, -15.1203575, -12.4987259, -1.5817308, 1.6863952
5: 8.2290249, 9.6807022, 8.2545176, 9.6398735, -1.0007670, 1.0343399
6: -4.7328625, -2.3143692, -4.7040586, -2.3734291, -1.5619869, 1.4815314
7: -15.7478466, -12.9592333, -15.7893782, -13.0131760, -1.8182282, 2.1953969
8: -0.8005610, 0.9433446, -0.7510371, 0.9524949, -1.1989489, 1.0797441
9: -6.6701612, -5.0296564, -6.6669064, -5.0131869, -1.5239248, 1.1434262

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6406490, upper bound: 0.6490706
time: 5.46 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6406490, upper bound: 0.6510503
time: 4.99 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.1792736, -10.5445099, -13.2840366, -10.5062513, -1.6417837, 1.5652127
1: -11.3019638, -8.4710150, -11.4454498, -8.4808769, -1.9268227, 2.0102696
2: -10.7013254, -8.5044823, -10.7834053, -8.5844460, -1.6302528, 1.8401673
3: -4.4191179, -2.3507521, -4.5502939, -2.3559453, -1.5353775, 1.5467656
4: -15.1293201, -12.5611839, -15.1077662, -12.5117950, -1.5853171, 1.6772943
5: 8.2291021, 9.6866627, 8.2472029, 9.6539249, -1.0142291, 1.0363431
6: -4.7521086, -2.3143733, -4.7492037, -2.3275783, -1.6223106, 1.4427271
7: -15.7501545, -12.9552126, -15.7946186, -13.0175314, -1.8141870, 2.2085567
8: -0.8006105, 0.9435668, -0.7508636, 0.9529798, -1.1993699, 1.0806515
9: -6.6692448, -5.0289845, -6.6630483, -5.0135283, -1.5269718, 1.1392655

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6521443, upper bound: 0.6510920
time: 3.82 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6488390, upper bound: 0.6510903
time: 6.08 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1307049, -10.5504866, -13.2338276, -10.5334206, -1.5180631, 1.5871561
1: -11.2909031, -8.4560642, -11.4258461, -8.4596052, -1.8753500, 1.9948034
2: -10.6618204, -8.5553007, -10.7431831, -8.5869207, -1.5686083, 1.7764182
3: -4.4226303, -2.3519435, -4.5749140, -2.3470745, -1.5663118, 1.5480886
4: -15.1365433, -12.5101528, -15.1465988, -12.4181728, -1.5870152, 1.7369361
5: 8.2241573, 9.6765308, 8.2302532, 9.6386633, -1.0096240, 1.0295012
6: -4.7152753, -2.3376703, -4.7305031, -2.3705282, -1.5278263, 1.4662068
7: -15.7377262, -12.9582272, -15.7815189, -13.0088482, -1.8295903, 2.1877060
8: -0.7974107, 0.9168944, -0.7833629, 0.9677305, -1.2108836, 1.0445035
9: -6.6717510, -5.0090761, -6.6731825, -4.9613705, -1.5476685, 1.1786079

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6379175, upper bound: 0.6603341
time: 3.95 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6379175, upper bound: 0.6509924
time: 4.86 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.1495314, -10.5504913, -13.2807884, -10.4994469, -1.5730677, 1.5778294
1: -11.2922621, -8.4564047, -11.4195824, -8.4772921, -1.8441887, 1.9944477
2: -10.6623688, -8.5531683, -10.7449856, -8.5799847, -1.5921521, 1.7690639
3: -4.4236050, -2.3542037, -4.5691385, -2.3606634, -1.5612741, 1.5412939
4: -15.1325474, -12.5098839, -15.1340017, -12.4312439, -1.5906024, 1.7278743
5: 8.2242327, 9.6824598, 8.2229252, 9.6527634, -1.0232646, 1.0314655
6: -4.7345695, -2.3376741, -4.7756162, -2.3246858, -1.5881686, 1.4274402
7: -15.7400074, -12.9541979, -15.7869215, -13.0132103, -1.8255243, 2.2011390
8: -0.7973878, 0.9171162, -0.7831311, 0.9682162, -1.2113848, 1.0454462
9: -6.6708355, -5.0083919, -6.6693244, -4.9617171, -1.5507236, 1.1745296

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6461074, upper bound: 0.6603765
time: 5.60 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6461074, upper bound: 0.6510367
time: 4.19 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.1605415, -10.5387220, -13.2347698, -10.5281677, -1.5868573, 1.5962627
1: -11.3006859, -8.4713554, -11.4272461, -8.4700565, -1.9673595, 1.9925594
2: -10.7007551, -8.4975376, -10.7571812, -8.5857487, -1.5862346, 1.8897357
3: -4.4289474, -2.3484936, -4.5753765, -2.3466854, -1.5733433, 1.5515780
4: -15.1333189, -12.5178509, -15.1466398, -12.4217949, -1.5913048, 1.7405248
5: 8.2116356, 9.6807041, 8.2300797, 9.6391220, -1.0258598, 1.0410922
6: -4.7454391, -2.3143673, -4.7307968, -2.3644786, -1.5908117, 1.4784718
7: -15.7478437, -12.9565420, -15.7827415, -13.0094862, -1.8398671, 2.1878686
8: -0.8162510, 0.9433463, -0.7888057, 0.9677904, -1.2345634, 1.0745497
9: -6.6701684, -5.0058818, -6.6717162, -4.9608417, -1.5483851, 1.2050388

Time for backsubstitution: 5.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6401354, upper bound: 0.6655113
time: 4.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6401354, upper bound: 0.6561695
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -13.1792870, -10.5387211, -13.2818222, -10.4941931, -1.6417952, 1.5869465
1: -11.3019676, -8.4716959, -11.4209652, -8.4877453, -1.9360847, 1.9922287
2: -10.7013254, -8.4957085, -10.7589817, -8.5788898, -1.6096387, 1.8824224
3: -4.4298735, -2.3507490, -4.5696025, -2.3602631, -1.5683298, 1.5447924
4: -15.1293278, -12.5175972, -15.1340466, -12.4348698, -1.5948930, 1.7314239
5: 8.2117119, 9.6866646, 8.2227621, 9.6532354, -1.0394557, 1.0430675
6: -4.7646885, -2.3143692, -4.7759342, -2.3186297, -1.6511364, 1.4396682
7: -15.7501535, -12.9525166, -15.7881727, -13.0138416, -1.8357773, 2.2012634
8: -0.8162985, 0.9435701, -0.7886314, 0.9682758, -1.2349834, 1.0754566
9: -6.6692529, -5.0052099, -6.6678562, -4.9611773, -1.5514321, 1.2009077

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 677
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 1779
type: A, layer: 3, pos: 1779
type: A, layer: 3, pos: 1835
type: B, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 677

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483254, upper bound: 0.6655531
time: 4.35 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6483254, upper bound: 0.6562137
time: 4.70 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.2361183, -10.5454826, -13.1306915, -10.5562706, -1.5654743, 1.5180309
1: -11.4503298, -8.4527369, -11.2909002, -8.4553823, -2.0128927, 1.8660350
2: -10.7676086, -8.5927334, -10.6618223, -8.5642605, -1.7341514, 1.5892272
3: -4.5554857, -2.3427744, -4.4118752, -2.3519464, -1.5500669, 1.5333500
4: -15.1203175, -12.4951077, -15.1365376, -12.5537252, -1.6828046, 1.5774565
5: 8.2546854, 9.6394339, 8.2415562, 9.6765289, -1.0227262, 0.9846144
6: -4.7037630, -2.3794785, -4.7026982, -2.3376732, -1.4694147, 1.4989982
7: -15.7882118, -13.0125408, -15.7377262, -12.9609308, -2.1953049, 1.8079138
8: -0.7455928, 0.9524357, -0.7817214, 0.9168925, -1.0497088, 1.1752696
9: -6.6683726, -5.0137129, -6.6717434, -5.0328526, -1.1170025, 1.5232081

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 2629
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6458967, upper bound: 0.6417386
time: 3.85 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6458967, upper bound: 0.6384332
time: 3.95 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.2830086, -10.5115061, -13.1495180, -10.5562754, -1.5561197, 1.5730076
1: -11.4440680, -8.4704237, -11.2922573, -8.4557257, -2.0124974, 1.8348925
2: -10.7694101, -8.5855942, -10.6623688, -8.5620155, -1.7268033, 1.6127732
3: -4.5498047, -2.3563483, -4.4128523, -2.3542085, -1.5432644, 1.5283170
4: -15.1077232, -12.5081692, -15.1325455, -12.5534697, -1.6737432, 1.5810423
5: 8.2473621, 9.6534691, 8.2416325, 9.6824579, -1.0247214, 0.9981171
6: -4.7488832, -2.3336358, -4.7219896, -2.3376770, -1.4306500, 1.5593400
7: -15.7934208, -13.0168991, -15.7400103, -12.9569006, -2.2084970, 1.8038998
8: -0.7453618, 0.9529214, -0.7816978, 0.9171162, -1.0506518, 1.1757715
9: -6.6645164, -5.0140653, -6.6708274, -5.0321670, -1.1128922, 1.5262632

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6459411, upper bound: 0.6499285
time: 3.92 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6459411, upper bound: 0.6466231
time: 3.95 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.2361183, -10.5454826, -13.2772141, -10.5113945, -1.4740112, 1.4648646
1: -11.4503298, -8.4527369, -11.4166174, -8.4609032, -1.8370419, 1.8664052
2: -10.7676086, -8.5927334, -10.7111168, -8.5750856, -1.6443572, 1.5816395
3: -4.5554857, -2.3427744, -4.5635281, -2.3452349, -1.3387222, 1.3419352
4: -15.1203175, -12.4951077, -15.1498919, -12.4747686, -1.5031395, 1.4797029
5: 8.2546854, 9.6394339, 8.2366257, 9.6387091, -0.9350415, 0.9519618
6: -4.7037630, -2.3794785, -4.7555871, -2.3436565, -1.3805990, 1.3755629
7: -15.7882118, -13.0125408, -15.7815304, -13.0041170, -1.8048091, 1.8030918
8: -0.7455928, 0.9524357, -0.7552834, 0.9699683, -1.0142562, 1.0069823
9: -6.6683726, -5.0137129, -6.6696639, -4.9782019, -1.1433485, 1.1262536

Time for backsubstitution: 5.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: B, layer: 3, pos: 1948
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: B, layer: 3, pos: 1983
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 2537
type: A, layer: 3, pos: 2537
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 2805
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 550
type: A, layer: 3, pos: 550
type: A, layer: 3, pos: 1194
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6361009, upper bound: 0.6384332
time: 3.90 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6380539, upper bound: 0.6384332
time: 4.00 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -13.2830086, -10.5115061, -13.2939949, -10.5113974, -1.4571319, 1.5241253
1: -11.4440680, -8.4704237, -11.4181032, -8.4663887, -1.8629403, 1.8351672
2: -10.7694101, -8.5855942, -10.7120199, -8.5734529, -1.6365290, 1.6079082
3: -4.5498047, -2.3563483, -4.5645313, -2.3494370, -1.3377361, 1.3324213
4: -15.1077232, -12.5081692, -15.1459036, -12.4745159, -1.4794431, 1.5021346
5: 8.2473621, 9.6534691, 8.2366991, 9.6435452, -0.9403397, 0.9615651
6: -4.7488832, -2.3336358, -4.7715750, -2.3436601, -1.3420415, 1.4576285
7: -15.7934208, -13.0168991, -15.7841167, -13.0051460, -1.8364382, 1.8150113
8: -0.7453618, 0.9529214, -0.7552962, 0.9702139, -1.0149860, 1.0088925
9: -6.6645164, -5.0140653, -6.6684752, -4.9774966, -1.1395273, 1.1266854

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 186
type: B, layer: 3, pos: 186
type: B, layer: 3, pos: 2530
type: A, layer: 3, pos: 2530
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 697
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: B, layer: 3, pos: 1507
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2468
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 2805
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1397
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 1397
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1194
type: A, layer: 3, pos: 429
type: B, layer: 3, pos: 429
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 772
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 2853
type: A, layer: 3, pos: 2853
type: A, layer: 3, pos: 949
type: B, layer: 3, pos: 949
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 3127
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 2831
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 2220
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 186

## Relational analysis of IS_A2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6380547, upper bound: 0.6499267
time: 6.97 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6380547, upper bound: 0.6466211
time: 7.41 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -13.2370558, -10.5402260, -13.1605310, -10.5445061, -1.5745564, 1.5868754
1: -11.4517336, -8.4631891, -11.3006821, -8.4706736, -2.0106387, 1.9580803
2: -10.7816067, -8.5915012, -10.7007542, -8.5064163, -1.8474755, 1.6068456
3: -4.5559764, -2.3423796, -4.4181957, -2.3484979, -1.5535579, 1.5403824
4: -15.1203575, -12.4987259, -15.1333160, -12.5614262, -1.6863952, 1.5817308
5: 8.2545176, 9.6398735, 8.2290249, 9.6807022, -1.0343399, 1.0007669
6: -4.7040586, -2.3734291, -4.7328625, -2.3143692, -1.4815311, 1.5619869
7: -15.7893782, -13.0131760, -15.7478466, -12.9592333, -2.1953974, 1.8182290
8: -0.7510371, 0.9524949, -0.8005610, 0.9433446, -1.0797439, 1.1989489
9: -6.6669064, -5.0131869, -6.6701612, -5.0296564, -1.1434259, 1.5239248

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 186
type: A, layer: 3, pos: 186
type: A, layer: 3, pos: 2530
type: B, layer: 3, pos: 2530
type: B, layer: 3, pos: 677
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 697
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 1836
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1948
type: B, layer: 3, pos: 415
type: A, layer: 3, pos: 1948
type: A, layer: 3, pos: 415
type: A, layer: 3, pos: 1983
type: B, layer: 3, pos: 1983
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1102
type: B, layer: 3, pos: 1102
type: A, layer: 3, pos: 1446
type: B, layer: 3, pos: 1446
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1507
type: B, layer: 3, pos: 1507
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2468
type: A, layer: 3, pos: 1922
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1922
type: A, layer: 3, pos: 2468
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2537
type: B, layer: 3, pos: 2537
type: B, layer: 3, pos: 2805
type: B, layer: 3, pos: 725
type: A, layer: 3, pos: 725
type: B, layer: 3, pos: 1111
type: A, layer: 3, pos: 2805
type: A, layer: 3, pos: 1397
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1397
type: A, layer: 3, pos: 2629
type: A, layer: 3, pos: 1111
type: B, layer: 3, pos: 425
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 425
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 961
type: B, layer: 3, pos: 961
type: A, layer: 3, pos: 1194
type: A, layer: 3, pos: 550
type: B, layer: 3, pos: 550
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 429
type: A, layer: 3, pos: 429
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 2629
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 772
type: A, layer: 3, pos: 410
type: B, layer: 3, pos: 1194
type: B, layer: 3, pos: 772
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 410
type: A, layer: 3, pos: 1402
type: B, layer: 3, pos: 1402
type: A, layer: 3, pos: 1829
type: B, layer: 3, pos: 1829
type: A, layer: 3, pos: 2853
type: B, layer: 3, pos: 2853
type: B, layer: 3, pos: 949
type: A, layer: 3, pos: 949
type: A, layer: 3, pos: 1412
type: B, layer: 3, pos: 3127
type: A, layer: 3, pos: 3127
type: B, layer: 3, pos: 1412
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 2831
type: A, layer: 3, pos: 2831
type: A, layer: 3, pos: 2914
type: B, layer: 3, pos: 2914
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 918
type: A, layer: 3, pos: 918
type: B, layer: 3, pos: 976
type: A, layer: 3, pos: 976
type: B, layer: 3, pos: 2220
type: A, layer: 3, pos: 2220
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 232
type: B, layer: 3, pos: 232
type: A, layer: 3, pos: 1779
type: B, layer: 3, pos: 1779
type: B, layer: 3, pos: 1835
type: A, layer: 3, pos: 1835

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 186

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6490707, upper bound: 0.6406511
time: 4.66 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6510503, upper bound: 0.6406512
time: 3.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.27 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6551738, upper bound: 0.6545893
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6571692, upper bound: 0.6545874
IS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6574457, upper bound: 0.6691948
IS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6574457, upper bound: 0.6659317
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6595617, upper bound: 0.6560062
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6616119, upper bound: 0.6560061
IS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6617135, upper bound: 0.6706271
IS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6617135, upper bound: 0.6673088
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6716563, upper bound: 0.6542133
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6716565, upper bound: 0.6542133
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6719335, upper bound: 0.6655319
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6719337, upper bound: 0.6655318
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6762605, upper bound: 0.6558192
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6762607, upper bound: 0.6558191
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6763628, upper bound: 0.6671214
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6763629, upper bound: 0.6671215
IS_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6417365, upper bound: 0.6458986
IS_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6384311, upper bound: 0.6458987
IS_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6499264, upper bound: 0.6459411
IS_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6466211, upper bound: 0.6459430
IS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6406490, upper bound: 0.6490706
IS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6406490, upper bound: 0.6510503
IS_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6521443, upper bound: 0.6510920
IS_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6488390, upper bound: 0.6510903
IS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6379175, upper bound: 0.6603341
IS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6379175, upper bound: 0.6509924
IS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6461074, upper bound: 0.6603765
IS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6461074, upper bound: 0.6510367
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6401354, upper bound: 0.6655113
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6401354, upper bound: 0.6561695
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6483254, upper bound: 0.6655531
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6483254, upper bound: 0.6562137
IS_A2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6458967, upper bound: 0.6417386
IS_A2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6458967, upper bound: 0.6384332
IS_A2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6459411, upper bound: 0.6499285
IS_A2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6459411, upper bound: 0.6466231
IS_A2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6361009, upper bound: 0.6384332
IS_A2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6380539, upper bound: 0.6384332
IS_A2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6380547, upper bound: 0.6499267
IS_A2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6380547, upper bound: 0.6466211
IS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6490707, upper bound: 0.6406511
IS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 5, lower bound: -0.6510503, upper bound: 0.6406512
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6509673, upper bound: 0.6576674
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6509666, upper bound: 0.6494776
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6509673, upper bound: 0.6576695
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6524895, upper bound: 0.6472611
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6524902, upper bound: 0.6554516
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6524895, upper bound: 0.6472611
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6524902, upper bound: 0.6554497
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6576666, upper bound: 0.6494795
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6576673, upper bound: 0.6576695
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6576666, upper bound: 0.6494795
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 5, lower bound: -0.6576673, upper bound: 0.6576695
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.0726536512374878
rel_dist={5: [-0.7059536409330516, 0.7059557061852892]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2414.06 seconds
