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
execution time: IAR + LP analysis = 13.17 + 31.94 = 45.11 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.89 seconds, max iter: 100)

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
Binary search time: 145.77 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_random_Z) starts
Time budget: 3409.12 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1479

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 697

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.0015028, upper bound: 0.9964605
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9964615, upper bound: 1.0015026
time: 4.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.68
Output dim: 5, lower bound: -1.0015028, upper bound: 0.9964605
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.68
Output dim: 5, lower bound: -0.9964615, upper bound: 1.0015026

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1230440, 2.1263335
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4697490, 2.4703960
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1608868, 2.1612737
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9598637, 1.9648561
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3206668, 2.3232961
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3203173, 1.3136952
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0786815, 2.0787497
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6789398, 2.6804223
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4574649, 1.4544401
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1446

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9850716, upper bound: 0.9940722
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9991131, upper bound: 0.9800457
time: 3.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1263342, 2.1230443
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4703960, 2.4697490
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1612740, 2.1608868
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9648561, 1.9598637
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3232961, 2.3206668
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3136952, 1.3203173
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0787497, 2.0786815
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6804218, 2.6789403
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4544399, 1.4574648
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 424

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9719225, upper bound: 0.9769638
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9719225, upper bound: 0.9769637
time: 5.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.12
Output dim: 5, lower bound: -0.9850716, upper bound: 0.9940722
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.12
Output dim: 5, lower bound: -0.9991131, upper bound: 0.9800457
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.12
Output dim: 5, lower bound: -0.9719225, upper bound: 0.9769638
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.12
Output dim: 5, lower bound: -0.9719225, upper bound: 0.9769637

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1178212, 2.1127553
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4647322, 2.4602587
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1554217, 2.1315219
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9577551, 1.9650426
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3129005, 2.3020976
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3016846, 1.3054091
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0455351, 2.0670633
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6660166, 2.6593046
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4565873, 1.4529920
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9695748, upper bound: 0.9744896
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9565003, upper bound: 0.9830164
time: 3.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1094661, 2.1211107
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4596119, 2.4653788
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1311350, 2.1558089
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9600506, 1.9627473
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2994680, 2.3155296
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3120308, 1.2950625
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0669951, 2.0456030
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6578226, 2.6674986
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4560161, 1.4535626
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1835

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9991125, upper bound: 0.9645520
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9836198, upper bound: 0.9800433
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1245527, 2.1271913
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4728179, 2.4683003
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1703286, 2.1580858
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9624910, 1.9585433
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3308244, 2.3147101
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3130190, 1.3244283
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0722108, 2.0782111
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6788177, 2.6767397
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4545934, 1.4570432
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9691800, upper bound: 0.9751910
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9701485, upper bound: 0.9742231
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1263342, 2.1212637
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4689474, 2.4697490
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1584725, 2.1608868
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9635353, 1.9598637
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3173394, 2.3206668
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3136952, 1.3196414
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0782790, 2.0786815
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6804218, 2.6773362
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4540184, 1.4574648
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 410

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9711439, upper bound: 0.9672179
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9661274, upper bound: 0.9761899
time: 5.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9695748, upper bound: 0.9744896
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9565003, upper bound: 0.9830164
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9991125, upper bound: 0.9645520
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9836198, upper bound: 0.9800433
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9691800, upper bound: 0.9751910
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9701485, upper bound: 0.9742231
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9711439, upper bound: 0.9672179
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.03
Output dim: 5, lower bound: -0.9661274, upper bound: 0.9761899

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1091456, 2.1087303
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4516444, 2.4523492
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1471944, 2.1276045
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9507637, 1.9564948
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3118963, 2.3021917
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2993085, 1.3005764
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0332112, 2.0578542
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6534843, 2.6514230
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4542041, 1.4488916
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9180136, upper bound: 0.9239199
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9180136, upper bound: 0.9239201
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1178212, 2.1040792
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4568224, 2.4602587
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1554217, 2.1232944
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9492073, 1.9650426
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3129005, 2.3010936
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3016846, 1.3030331
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0363259, 2.0670633
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6581354, 2.6593046
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4524865, 1.4529920
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2831

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 976

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9532096, upper bound: 0.9660178
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9395018, upper bound: 0.9797233
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1084843, 2.1192660
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4589624, 2.4651382
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1313739, 2.1563249
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9585724, 1.9620776
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2997198, 2.3152258
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3118881, 1.2947282
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0667739, 2.0453174
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6572351, 2.6672182
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4553933, 1.4532812
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 772

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9930810, upper bound: 0.9625450
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9971073, upper bound: 0.9585179
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1076212, 2.1201296
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4593711, 2.4647298
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1316509, 2.1560478
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9593806, 1.9612694
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2991643, 2.3157811
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3116966, 1.2949196
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0667095, 2.0453815
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6575422, 2.6669116
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4557347, 1.4529397
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9697258, upper bound: 0.9566770
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9590130, upper bound: 0.9677962
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1056743, 2.1156585
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4756899, 2.4699109
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1668372, 2.1551237
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9464669, 1.9460018
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3255711, 2.2915823
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3080184, 1.3207898
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0704041, 2.0759907
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6860933, 2.6804070
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4258587, 1.4466395
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 415

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9389540, upper bound: 0.9450801
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9389540, upper bound: 0.9450801
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1130204, 2.1083124
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4744287, 2.4702013
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1677403, 2.1545939
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9503789, 1.9425187
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3076968, 2.3098013
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3093803, 1.3194275
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0699902, 2.0764046
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6824856, 2.6837482
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4429400, 1.4283085
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9696620, upper bound: 0.9445205
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9410541, upper bound: 0.9737366
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1240282, 2.1240075
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4569030, 2.4574325
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1555748, 2.1593266
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9595876, 1.9554806
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3178630, 2.3212609
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3128793, 1.3171809
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0800772, 2.0791655
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6725283, 2.6756287
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4536114, 1.4570334
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2530

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2853

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9702609, upper bound: 0.9637864
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9698464, upper bound: 0.9662900
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1295834, 2.1189582
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4568653, 2.4577060
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1569128, 2.1579885
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9591522, 1.9569914
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3179331, 2.3211906
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3112350, 1.3189039
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0787640, 2.0804789
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6787138, 2.6694436
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4535871, 1.4570581
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9544214, upper bound: 0.9630613
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9529972, upper bound: 0.9644838
time: 5.25 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9180136, upper bound: 0.9239199
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9180136, upper bound: 0.9239201
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9532096, upper bound: 0.9660178
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9395018, upper bound: 0.9797233
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9930810, upper bound: 0.9625450
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9971073, upper bound: 0.9585179
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9697258, upper bound: 0.9566770
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9590130, upper bound: 0.9677962
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9389540, upper bound: 0.9450801
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9389540, upper bound: 0.9450801
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9696620, upper bound: 0.9445205
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9410541, upper bound: 0.9737366
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9702609, upper bound: 0.9637864
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9698464, upper bound: 0.9662900
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9544214, upper bound: 0.9630613
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 5, lower bound: -0.9529972, upper bound: 0.9644838

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1085744, 2.1147399
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4509573, 2.4521878
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1462932, 2.1293914
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9498205, 1.9403672
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3127828, 2.2993231
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2992828, 1.2989998
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0311713, 2.0646772
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6609106, 2.6483059
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4522071, 1.4453971
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8374788, upper bound: 0.8433849
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8374788, upper bound: 0.8433849
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1091456, 2.1081593
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4514828, 2.4523492
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1471944, 2.1267030
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9507637, 1.9555514
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3090281, 2.3021917
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2993085, 1.3005505
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0332112, 2.0558140
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6503677, 2.6514230
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4542041, 1.4468943
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 3127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9052872, upper bound: 0.9142606
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9086000, upper bound: 0.9074170
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1178226, 2.1040704
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4567690, 2.4602423
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1527612, 2.1220007
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9489441, 1.9648056
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3088756, 2.2985249
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3015350, 1.3024505
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0360498, 2.0663986
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6525440, 2.6563659
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4515910, 1.4516015
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1948

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9497838, upper bound: 0.9548961
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9379641, upper bound: 0.9622483
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1178236, 2.1040802
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4568062, 2.4602587
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1541283, 2.1223085
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9489708, 1.9649043
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3109798, 2.2970693
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3011016, 1.3027112
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0356607, 2.0666583
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6555624, 2.6537137
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4519906, 1.4520960
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 3127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2914

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9301833, upper bound: 0.9679641
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9277416, upper bound: 0.9704319
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1173258, 2.1202753
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4223061, 2.4370680
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1127224, 2.1324792
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9561100, 1.9635830
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2460012, 2.2624135
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3113358, 1.2945085
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.9777527, 1.9304905
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6520739, 2.6640635
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4362946, 1.4227848
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9909510, upper bound: 0.9609552
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9914908, upper bound: 0.9553520
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1094942, 2.1281071
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4308925, 2.4284816
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1075282, 2.1376731
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9600778, 1.9596148
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2469072, 2.2615073
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3116684, 1.2941761
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.9519463, 1.9562964
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6540804, 2.6620564
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4248972, 1.4341817
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 963

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9946231, upper bound: 0.9560758
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9950691, upper bound: 0.9552862
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0989447, 2.1161044
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4462829, 2.4568200
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1234241, 2.1521313
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9523902, 1.9527225
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2981601, 2.3158741
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3093206, 1.2896612
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0543861, 2.0361724
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6450090, 2.6590290
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4533520, 1.4488394
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 424

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 425

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9620820, upper bound: 0.9565587
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9696070, upper bound: 0.9490344
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1076212, 2.1114533
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4514608, 2.4647298
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1316509, 2.1478209
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9508338, 1.9612694
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2991643, 2.3147769
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3116966, 1.2925438
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0575004, 2.0453815
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6496592, 2.6669116
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4516349, 1.4529397
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9590087, upper bound: 0.9677457
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9550368, upper bound: 0.9667434
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1038771, 2.1149051
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4699655, 2.4664378
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1590414, 2.1425619
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9437866, 1.9420078
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3236594, 2.2861495
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3050396, 1.3164520
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0717216, 2.0722456
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6825953, 2.6780391
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4252622, 1.4461370
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1446

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9364412, upper bound: 0.9432836
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9371577, upper bound: 0.9425671
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1049209, 2.1156585
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4722171, 2.4699109
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1542749, 2.1551237
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9424725, 1.9460018
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3255711, 2.2896709
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3036807, 1.3207898
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0666595, 2.0759907
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6837244, 2.6804070
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4253561, 1.4466395
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2537

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9235677, upper bound: 0.9299238
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9235677, upper bound: 0.9299239
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1048918, 2.1166985
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4692473, 2.4672914
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1667228, 2.1556396
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9499826, 1.9427714
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2895517, 2.2973709
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3097230, 1.3126804
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0740209, 2.0854905
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6842079, 2.6910696
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4426172, 1.4282529
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1446

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 677

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9394443, upper bound: 0.9339146
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9592209, upper bound: 0.9142832
time: 3.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1214066, 2.1001842
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4715190, 2.4650197
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1687856, 2.1535764
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9506311, 1.9421227
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2952666, 2.2916565
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3026334, 1.3197702
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0790763, 2.0804350
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6898079, 2.6854692
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4428847, 1.4279854
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1983

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9363542, upper bound: 0.9603605
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9276806, upper bound: 0.9690275
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1238446, 2.1254268
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4487529, 2.4526739
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1550007, 2.1598282
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9574828, 1.9527776
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3059077, 2.3124380
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3127805, 1.3163729
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0765042, 2.0765903
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6639881, 2.6700106
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4482009, 1.4481393
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1412

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613236, upper bound: 0.9548469
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9613236, upper bound: 0.9548468
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1254468, 2.1239712
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4521451, 2.4492815
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1560769, 2.1587524
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9569926, 1.9533758
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3090396, 2.3093064
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3123366, 1.3170819
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0775018, 2.0755925
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6669121, 2.6670871
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4447176, 1.4516227
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9692877, upper bound: 0.9457376
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9492949, upper bound: 0.9657301
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1257100, 2.1163378
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.2853847, 2.3173745
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1495008, 2.1774244
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9534545, 1.9436738
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3039055, 2.2962523
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2918456, 1.2996964
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0786977, 2.0804367
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6540298, 2.6478105
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3794630, 1.4111549
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1836

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9476483, upper bound: 0.9368735
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9282340, upper bound: 0.9562873
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1270823, 2.1172054
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3660626, 2.2862163
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1616459, 2.1516275
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9457898, 1.9608555
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3211274, 2.3075602
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2922871, 1.3125942
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0787439, 2.0804129
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6659737, 2.6450038
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4176927, 1.3831379
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9332558, upper bound: 0.9463208
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9360592, upper bound: 0.9515039
time: 3.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.8374788, upper bound: 0.8433849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.8374788, upper bound: 0.8433849
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9052872, upper bound: 0.9142606
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9086000, upper bound: 0.9074170
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9497838, upper bound: 0.9548961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9379641, upper bound: 0.9622483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9301833, upper bound: 0.9679641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9277416, upper bound: 0.9704319
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9909510, upper bound: 0.9609552
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9914908, upper bound: 0.9553520
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9946231, upper bound: 0.9560758
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9950691, upper bound: 0.9552862
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9620820, upper bound: 0.9565587
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9696070, upper bound: 0.9490344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9590087, upper bound: 0.9677457
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9550368, upper bound: 0.9667434
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9364412, upper bound: 0.9432836
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9371577, upper bound: 0.9425671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9235677, upper bound: 0.9299238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9235677, upper bound: 0.9299239
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9394443, upper bound: 0.9339146
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9592209, upper bound: 0.9142832
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9363542, upper bound: 0.9603605
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9276806, upper bound: 0.9690275
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9613236, upper bound: 0.9548469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9613236, upper bound: 0.9548468
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9692877, upper bound: 0.9457376
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9492949, upper bound: 0.9657301
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9476483, upper bound: 0.9368735
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9282340, upper bound: 0.9562873
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9332558, upper bound: 0.9463208
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.58
Output dim: 5, lower bound: -0.9360592, upper bound: 0.9515039

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1085153, 2.1224508
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4574399, 2.4518332
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1679618, 2.1287386
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -2.0438175, 1.9376199
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3118882, 2.3400736
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2989748, 1.3026760
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0310073, 2.0647187
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6933279, 2.6478162
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4515615, 1.4425473
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 410

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8370792, upper bound: 0.8420126
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8292920, upper bound: 0.8424512
time: 3.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1085744, 2.1146810
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4506025, 2.4521878
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1456401, 2.1293914
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9470730, 1.9403672
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3127828, 2.2984285
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2992828, 1.2986920
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0311713, 2.0645132
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6604214, 2.6483059
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4522071, 1.4447513
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8369176, upper bound: 0.8228346
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8169191, upper bound: 0.8428262
time: 3.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1165996, 2.0880859
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4311485, 2.4130619
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1426163, 2.1165154
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9507570, 1.9451122
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2877965, 2.2891240
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2932386, 1.3033054
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0178175, 2.0452714
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6436720, 2.6386433
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4255047, 1.4251478
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1836

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8985660, upper bound: 0.8880656
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8789037, upper bound: 0.9074886
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0890718, 2.1081593
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4121962, 2.4523492
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1471944, 2.1221242
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9403248, 1.9555514
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2959604, 2.3021917
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2993085, 1.2944809
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0226688, 2.0558140
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6375885, 2.6514230
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4542041, 1.4181943
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8212122, upper bound: 0.8271348
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8212122, upper bound: 0.8271348
time: 3.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1089811, 2.0831556
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4361768, 2.4496114
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1318288, 2.1033671
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9434667, 1.9609995
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2219267, 2.2447839
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2997924, 1.2956938
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0291920, 2.0604086
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.4882579, 2.5331917
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4457905, 1.4454041
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 961

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9492972, upper bound: 0.9240284
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9204814, upper bound: 0.9544082
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0969086, 2.0974855
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4471827, 2.4396493
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1341267, 2.1144941
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9495673, 1.9593277
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2739644, 2.2115741
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2947789, 1.3012565
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0300608, 2.0644608
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.5763755, 2.4920812
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4495537, 1.4458015
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9364079, upper bound: 0.9559074
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9304927, upper bound: 0.9602702
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1158319, 2.1037574
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4438276, 2.4578681
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1524656, 2.1208158
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9437609, 1.9837055
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3144383, 2.2960315
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3012209, 1.3027017
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0265436, 2.0679820
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6564026, 2.6528487
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4507308, 1.4513701
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2537

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9165749, upper bound: 0.9577131
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9166025, upper bound: 0.9575562
time: 4.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1175013, 2.1040802
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4544148, 2.4602587
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1526358, 2.1223085
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9489708, 1.9596949
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3099418, 2.2970693
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3010926, 1.3027112
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0356607, 2.0575411
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6546965, 2.6537137
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4512649, 1.4520960
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1412

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 949

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9274301, upper bound: 0.9485434
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9173836, upper bound: 0.9691823
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0970573, 2.1073527
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4195752, 2.4340467
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1129198, 2.1344628
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9391041, 1.9523714
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2159834, 2.2141769
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3013821, 1.2859166
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.9747586, 1.9270830
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6595097, 2.6678920
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4082739, 1.4130955
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9905516, upper bound: 0.9550546
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9873282, upper bound: 0.9605168
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1044035, 2.1000071
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4192853, 2.4336083
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1134496, 2.1326771
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9425869, 1.9465773
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.1977644, 2.2252712
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3013821, 1.2845547
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.9734392, 1.9274967
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6559029, 2.6714983
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4268582, 1.3947644
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9576137, upper bound: 0.9252671
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9576137, upper bound: 0.9252667
time: 5.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1088200, 2.1271989
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4298897, 2.4287088
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1068501, 2.1370542
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9596252, 1.9590094
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2463908, 2.2608848
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3115225, 1.2940284
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.9516239, 1.9561083
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6538334, 2.6617789
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4237843, 1.4337814
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9923215, upper bound: 0.9559096
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9944216, upper bound: 0.9533348
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1094942, 2.1274328
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4308925, 2.4274790
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1075282, 2.1369946
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9600778, 1.9591622
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2462850, 2.2615073
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3116684, 1.2940302
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.9519463, 1.9559743
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6540804, 2.6618090
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4244967, 1.4341817
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9818621, upper bound: 0.9388213
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9766811, upper bound: 0.9406478
time: 3.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1019073, 2.1141429
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4461107, 2.4565001
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1179349, 2.1450331
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9511261, 1.9530473
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2980080, 2.3151569
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3054314, 1.2876031
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0542741, 2.0353997
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6449804, 2.6580648
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4552901, 1.4513421
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9617147, upper bound: 0.9523302
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9587044, upper bound: 0.9561383
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0969834, 2.1190667
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4459629, 2.4566476
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1163256, 2.1466427
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9527149, 1.9514585
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2974424, 2.3157220
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3072624, 1.2857720
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0536132, 2.0360608
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6440449, 2.6590004
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4558547, 1.4507771
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2537

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9571249, upper bound: 0.9365534
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9571249, upper bound: 0.9365534
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1081667, 2.1121492
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4506154, 2.4632537
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1327796, 2.1484630
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9509616, 1.9607575
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2981787, 2.3144140
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3120835, 1.2930570
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0572248, 2.0465989
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6490870, 2.6664648
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4498327, 1.4502023
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1983

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9542528, upper bound: 0.9543699
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9519518, upper bound: 0.9630435
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1083174, 2.1119986
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4499860, 2.4638841
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1322923, 2.1489496
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9503207, 1.9613986
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2988009, 2.3137913
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3122098, 1.2929308
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0587254, 2.0451059
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6492147, 2.6663375
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4488971, 1.4511379
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1507

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9498496, upper bound: 0.9636699
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9519627, upper bound: 0.9615559
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1038380, 2.1148074
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4697275, 2.4651206
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1584511, 2.1416183
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9439516, 1.9419253
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3236599, 2.2861426
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3044761, 1.3161530
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0716448, 2.0722058
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6824789, 2.6774526
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4250612, 1.4459913
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1446

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 429

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9310172, upper bound: 0.9399403
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9333328, upper bound: 0.9333906
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1037793, 2.1149051
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4699655, 2.4661999
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1590414, 2.1419716
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9437037, 1.9420078
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3236523, 2.2861495
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3047403, 1.3164520
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0716820, 2.0722456
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6825953, 2.6779227
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4252622, 1.4459357
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1402

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9360521, upper bound: 0.9398040
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9345206, upper bound: 0.9412077
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1048932, 2.1149445
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4779401, 2.4689951
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1473327, 2.1504235
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9408789, 1.9454372
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3294373, 2.2886779
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3024700, 1.3222928
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0588241, 2.0155685
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6890678, 2.6792054
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4145463, 1.4514000
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9033567, upper bound: 0.9096832
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9033567, upper bound: 0.9096832
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1049209, 2.1156306
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4713011, 2.4699109
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1495743, 2.1551237
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9419079, 1.9460018
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3245778, 2.2896709
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3036807, 1.3195789
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0666595, 2.0681558
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6825209, 2.6804070
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4253561, 1.4358296
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 429

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9171896, upper bound: 0.9254955
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9191495, upper bound: 0.9235467
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0874915, 2.0938554
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4674802, 2.4791503
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0940380, 2.0867772
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8983107, 1.9049876
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2346530, 2.2014122
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2685127, 1.2975146
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0277724, 2.0525773
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6925097, 2.6872239
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3810079, 1.3852699
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 976

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9361206, upper bound: 0.9159623
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9224353, upper bound: 0.9295367
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0820494, 2.0987844
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4811058, 2.4640424
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0978599, 2.0817237
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9120569, 1.8910992
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.1935930, 2.2391050
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2907271, 1.2714702
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0403948, 2.0392418
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6803608, 2.6981306
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3988941, 1.3666434
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9386603, upper bound: 0.8941501
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9386603, upper bound: 0.8941501
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1186156, 2.0964749
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4496450, 2.4705942
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1578922, 2.1412489
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9181700, 1.9530146
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2915635, 2.2852731
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3020713, 1.3147244
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0768719, 2.0786664
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6868515, 2.6800075
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4390652, 1.4269634
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2930

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9099633, upper bound: 0.9332042
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9099633, upper bound: 0.9332042
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1214066, 2.0973940
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4715190, 2.4431458
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1687856, 2.1426830
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9506311, 1.9096618
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2952666, 2.2879531
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2975874, 1.3197702
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0773077, 2.0804350
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6898079, 2.6825132
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4428847, 1.4241661
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1507

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9243822, upper bound: 0.9659536
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9249823, upper bound: 0.9638442
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1238327, 2.1256230
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4486732, 2.4526377
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1549621, 2.1598449
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9574957, 1.9527407
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3050880, 2.3124115
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3127290, 1.3163321
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0764613, 2.0767412
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6637259, 2.6699581
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4481604, 1.4481087
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1479

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9321736, upper bound: 0.9203208
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9321736, upper bound: 0.9203226
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1238446, 2.1254144
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4487529, 2.4525945
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1550007, 2.1597896
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9574461, 1.9527776
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3058815, 2.3124380
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3127805, 1.3163214
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0765042, 2.0765479
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6639366, 2.6700106
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4482009, 1.4480989
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 677

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2831

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9610159, upper bound: 0.9455852
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9520596, upper bound: 0.9545395
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1239634, 2.1239460
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4471693, 2.4491169
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1576085, 2.1703439
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9566460, 1.9533033
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3084836, 2.3084793
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3185236, 1.3125529
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0679970, 2.0718651
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6655941, 2.6718283
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4436049, 1.4524739
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9530805, upper bound: 0.9254843
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9530805, upper bound: 0.9254843
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1254215, 2.1224873
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4519806, 2.4443054
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1676679, 2.1602845
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9569206, 1.9530287
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3082132, 2.3087499
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3078077, 1.3232689
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0737753, 2.0660868
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6716537, 2.6657691
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4455690, 1.4505098
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9375887, upper bound: 0.9525984
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9361647, upper bound: 0.9540253
time: 4.46 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8370792, upper bound: 0.8420126
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8292920, upper bound: 0.8424512
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8369176, upper bound: 0.8228346
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8169191, upper bound: 0.8428262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8985660, upper bound: 0.8880656
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8789037, upper bound: 0.9074886
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8212122, upper bound: 0.8271348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.8212122, upper bound: 0.8271348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9492972, upper bound: 0.9240284
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9204814, upper bound: 0.9544082
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9364079, upper bound: 0.9559074
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9304927, upper bound: 0.9602702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9165749, upper bound: 0.9577131
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9166025, upper bound: 0.9575562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9274301, upper bound: 0.9485434
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9173836, upper bound: 0.9691823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9905516, upper bound: 0.9550546
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9873282, upper bound: 0.9605168
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9576137, upper bound: 0.9252671
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9576137, upper bound: 0.9252667
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9923215, upper bound: 0.9559096
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9944216, upper bound: 0.9533348
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9818621, upper bound: 0.9388213
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9766811, upper bound: 0.9406478
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9617147, upper bound: 0.9523302
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9587044, upper bound: 0.9561383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9571249, upper bound: 0.9365534
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9571249, upper bound: 0.9365534
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9542528, upper bound: 0.9543699
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9519518, upper bound: 0.9630435
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9498496, upper bound: 0.9636699
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9519627, upper bound: 0.9615559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9310172, upper bound: 0.9399403
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9333328, upper bound: 0.9333906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9360521, upper bound: 0.9398040
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9345206, upper bound: 0.9412077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9033567, upper bound: 0.9096832
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9033567, upper bound: 0.9096832
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9171896, upper bound: 0.9254955
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9191495, upper bound: 0.9235467
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9361206, upper bound: 0.9159623
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9224353, upper bound: 0.9295367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9386603, upper bound: 0.8941501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9386603, upper bound: 0.8941501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9099633, upper bound: 0.9332042
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9099633, upper bound: 0.9332042
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9243822, upper bound: 0.9659536
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9249823, upper bound: 0.9638442
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9321736, upper bound: 0.9203208
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9321736, upper bound: 0.9203226
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9610159, upper bound: 0.9455852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9520596, upper bound: 0.9545395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9530805, upper bound: 0.9254843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9530805, upper bound: 0.9254843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9375887, upper bound: 0.9525984
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.82
Output dim: 5, lower bound: -0.9361647, upper bound: 0.9540253
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 5, lower bound: -0.9476483, upper bound: 0.9368735
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 5, lower bound: -0.9282340, upper bound: 0.9562873
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 5, lower bound: -0.9332558, upper bound: 0.9463208
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 5, lower bound: -0.9360592, upper bound: 0.9515039
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.3380417823791504
rel_dist={5: [-1.0100506396824613, 1.010049981815225]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 425

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2216

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7787311, upper bound: 0.7793377
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7793377, upper bound: 0.7787309
time: 4.22 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.44
Output dim: 5, lower bound: -0.7787311, upper bound: 0.7793377
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.44
Output dim: 5, lower bound: -0.7793377, upper bound: 0.7787309

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7695422, 1.7695060
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0661564, 2.0635707
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8669825, 1.8642974
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6923399, 1.7011063
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9170332, 1.9147902
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1361872, 1.1368109
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7535911, 1.7427227
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3095331, 2.3095331
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2416310, 1.2418278
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6355152, 1.6352506

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7616655, upper bound: 0.7622695
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7616655, upper bound: 0.7622692
time: 5.03 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7700553, 1.7695422
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0635710, 2.0679142
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8707552, 1.8669825
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.7011061, 1.7071259
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9147902, 1.9186454
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1390005, 1.1361872
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7427225, 1.7544298
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3095331, 2.3095465
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2417350, 1.2416310
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6392679, 1.6355152

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 677

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7721885, upper bound: 0.7767393
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7776100, upper bound: 0.7713712
time: 5.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 5, lower bound: -0.7616655, upper bound: 0.7622695
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 5, lower bound: -0.7616655, upper bound: 0.7622692
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 5, lower bound: -0.7721885, upper bound: 0.7767393
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 5, lower bound: -0.7776100, upper bound: 0.7713712

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7661653, 1.7484479
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0721097, 2.0513036
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8649158, 1.8629353
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6932249, 1.6912920
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9139781, 1.9059234
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1306481, 1.1326196
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7432837, 1.7346737
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3058357, 2.3067207
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2362266, 1.2343130
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6238837, 1.6171741

Time for backsubstitution: 4.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2914

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7482985, upper bound: 0.7488324
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7482280, upper bound: 0.7489022
time: 6.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7484841, 1.7695060
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0538888, 2.0635707
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8669825, 1.8622310
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6825256, 1.7011063
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9170332, 1.9117351
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1319959, 1.1368109
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7455420, 1.7427227
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3095331, 2.3058367
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2416310, 1.2364235
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6355152, 1.6236186

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 677

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7079043, upper bound: 0.7085080
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7079043, upper bound: 0.7085080
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7643585, 1.7632642
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0594797, 2.0662305
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8579798, 1.8515494
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6775160, 1.6852219
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9115787, 1.9173863
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1361029, 1.1342242
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7216983, 1.7253735
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2909355, 2.2885876
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2296841, 1.2285469
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6385903, 1.6354556

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1948

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7680866, upper bound: 0.7646767
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7601205, upper bound: 0.7726371
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7636662, 1.7639573
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0619097, 2.0638006
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8561983, 1.8533309
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6774759, 1.6852622
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9125328, 1.9164319
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1369977, 1.1333293
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7143011, 1.7327707
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2885742, 2.2909479
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2285914, 1.2296395
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6392102, 1.6348357

Time for backsubstitution: 4.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7699165, upper bound: 0.7636779
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7699149, upper bound: 0.7636807
time: 4.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7482985, upper bound: 0.7488324
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7482280, upper bound: 0.7489022
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7079043, upper bound: 0.7085080
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7079043, upper bound: 0.7085080
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7680866, upper bound: 0.7646767
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7601205, upper bound: 0.7726371
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7699165, upper bound: 0.7636779
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.39
Output dim: 5, lower bound: -0.7699149, upper bound: 0.7636807

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7643995, 1.7481272
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0605898, 2.0485651
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8633175, 1.8614433
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6880155, 1.7010894
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9157515, 1.9048862
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1307192, 1.1326106
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7341666, 1.7320824
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3060369, 2.3058558
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2351675, 1.2335875
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6214924, 1.6102257

Time for backsubstitution: 4.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7409664, upper bound: 0.7471136
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7463296, upper bound: 0.7417125
time: 5.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7658443, 1.7484479
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0693712, 2.0513036
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8634243, 1.8629353
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6932249, 1.6860828
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9129410, 1.9059234
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1306391, 1.1326196
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7432837, 1.7255569
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3049717, 2.3067207
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2355013, 1.2343130
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6169348, 1.6171741

Time for backsubstitution: 4.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 425

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7176507, upper bound: 0.7183255
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7176507, upper bound: 0.7183255
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7469254, 1.7693543
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0530095, 2.0651484
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8668633, 1.8620238
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6827374, 1.7010179
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9168162, 1.9141047
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1304367, 1.1366100
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7422786, 1.7424855
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3143806, 2.3055429
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2414393, 1.2376497
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6353068, 1.6246166

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2530

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7018276, upper bound: 0.7024523
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7018464, upper bound: 0.7024297
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7483134, 1.7695060
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0538888, 2.0626910
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8669825, 1.8621123
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6824379, 1.7011063
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9170332, 1.9115181
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1317952, 1.1368109
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7453041, 1.7427227
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3092394, 2.3058367
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2416310, 1.2362317
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6355152, 1.6234097

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2831

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 976

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7055719, upper bound: 0.6969062
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6963021, upper bound: 0.7061756
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7509918, 1.7423518
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0388975, 2.0518727
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8363659, 1.8313699
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6721048, 1.6808553
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8243876, 1.8500876
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1324542, 1.1274422
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7151146, 1.7193320
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1263938, 2.1494050
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2233129, 1.2219268
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6233439, 1.6269374

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 772

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7631351, upper bound: 0.7622974
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7656671, upper bound: 0.7604383
time: 5.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7434464, 1.7498975
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0451236, 2.0456467
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8377993, 1.8299360
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6731496, 1.6798105
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8442802, 1.8301950
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1293209, 1.1305755
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7156572, 1.7187898
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1517520, 2.1240463
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2230644, 1.2221754
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6300731, 1.6202087

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 725

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7583740, upper bound: 0.7660923
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7535257, upper bound: 0.7709068
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7628126, 1.7620816
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0643315, 2.0625329
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8532457, 1.8531914
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6773934, 1.6850984
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9114442, 1.9149396
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1369629, 1.1333083
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7145748, 1.7326195
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2929735, 2.2890339
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2285457, 1.2293780
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6389985, 1.6347942

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7696124, upper bound: 0.7459280
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7521648, upper bound: 0.7633740
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7636662, 1.7631049
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0606427, 2.0638006
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8561983, 1.8503792
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6774759, 1.6851795
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9110413, 1.9164319
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1369977, 1.1332946
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7141504, 1.7327707
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2866592, 2.2909479
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2285914, 1.2295933
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6392102, 1.6346240

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7696036, upper bound: 0.7613285
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7675634, upper bound: 0.7633703
time: 4.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7409664, upper bound: 0.7471136
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7463296, upper bound: 0.7417125
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7176507, upper bound: 0.7183255
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7176507, upper bound: 0.7183255
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7018276, upper bound: 0.7024523
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7018464, upper bound: 0.7024297
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7055719, upper bound: 0.6969062
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.6963021, upper bound: 0.7061756
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7631351, upper bound: 0.7622974
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7656671, upper bound: 0.7604383
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7583740, upper bound: 0.7660923
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7535257, upper bound: 0.7709068
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7696124, upper bound: 0.7459280
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7521648, upper bound: 0.7633740
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7696036, upper bound: 0.7613285
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 5, lower bound: -0.7675634, upper bound: 0.7633703

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7588143, 1.7418489
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0564995, 2.0469046
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8496652, 1.8460100
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6644254, 1.6774590
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9125414, 1.9026303
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1278614, 1.1306486
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7126327, 1.7036600
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2874374, 2.2848949
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2231750, 1.2205715
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6208134, 1.6101661

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7333378, upper bound: 0.7352690
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7302629, upper bound: 0.7379446
time: 3.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7581210, 1.7425432
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0588398, 2.0444746
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8478842, 1.8476629
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6643362, 1.6774993
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9134798, 1.9016759
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1287562, 1.1297526
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7057443, 1.7110572
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2850761, 2.2872553
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2220824, 1.2215955
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6214328, 1.6095448

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1507

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7426788, upper bound: 0.7393686
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7439834, upper bound: 0.7380641
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7652206, 1.7519369
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0688820, 2.0511434
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8625274, 1.8637183
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6922941, 1.6756611
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9110680, 1.9026313
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1305509, 1.1315624
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7412434, 1.7295489
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3088713, 2.3036041
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2335033, 1.2313741
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6175818, 1.6145277

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1983

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7148552, upper bound: 0.7105689
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7095009, upper bound: 0.7155619
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7658443, 1.7478242
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0692101, 2.0513036
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8634243, 1.8620381
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6932249, 1.6851511
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9096484, 1.9059234
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1306391, 1.1325315
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7432837, 1.7235162
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3018541, 2.3067207
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2355013, 1.2323153
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6142874, 1.6171741

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 425

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7103206, upper bound: 0.7166058
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7156799, upper bound: 0.7112068
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7212901, 1.7462499
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0450392, 2.0576496
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8667026, 1.8546934
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6588068, 1.6637030
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8972511, 1.8934135
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1153139, 1.1235158
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6809611, 1.7295620
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3009043, 2.2714281
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1538372, 1.2076378
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6235566, 1.5771832

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1194

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6988046, upper bound: 0.6884089
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6877246, upper bound: 0.6992446
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7285929, 1.7365971
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0463028, 2.0518227
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8595328, 1.8618631
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6471262, 1.6760120
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8948846, 1.8953681
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1177255, 1.1185852
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7293491, 1.6811686
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2802649, 2.2920589
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2105212, 1.1509525
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5878735, 1.6099787

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7015695, upper bound: 0.6999673
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6993819, upper bound: 0.7021532
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7483130, 1.7695003
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0538492, 2.0626750
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8648338, 1.8608184
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6821833, 1.7008693
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9130073, 1.9084029
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1314832, 1.1362276
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7448826, 1.7420580
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3036499, 2.3019042
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2407346, 1.2350261
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6353836, 1.6233187

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1397

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6748391, upper bound: 0.6659567
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6748391, upper bound: 0.6659567
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7483072, 1.7695065
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0538726, 2.0626519
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8656883, 1.8599639
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6822000, 1.7008526
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9139171, 1.9074931
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1312124, 1.1364986
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7446394, 1.7423012
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3053074, 2.3002467
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2404256, 1.2353351
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6354241, 1.6232781

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 550

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6923341, upper bound: 0.7053561
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6952175, upper bound: 0.7020444
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7604265, 1.7468908
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0013456, 2.0194554
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8168731, 1.8086309
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6690202, 1.6800878
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7643557, 1.7886167
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1318648, 1.1270791
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6244216, 1.6125102
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1108027, 2.1342587
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2000697, 1.1915604
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5544457, 1.5660815

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1983

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7603544, upper bound: 0.7527944
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7536113, upper bound: 0.7595194
time: 6.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7555218, 1.7517858
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0067124, 2.0143230
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8134236, 1.8118770
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6715002, 1.6777706
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7649221, 1.7900553
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1320910, 1.1268713
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6093974, 1.6286390
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1120577, 2.1338139
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1929464, 1.1986835
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5624881, 1.5603294

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1836

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7594843, upper bound: 0.7548790
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7595130, upper bound: 0.7548785
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7442937, 1.7514038
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0377598, 2.0385623
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8340063, 1.8307672
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6684995, 1.6722538
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8012390, 1.7916999
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1286373, 1.1274660
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7076406, 1.7064188
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1466026, 2.1193585
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2116661, 1.2114983
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6354017, 1.6278610

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7506189, upper bound: 0.7539642
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7476512, upper bound: 0.7569611
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7449536, 1.7507443
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0380373, 2.0382848
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8386307, 1.8261428
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6655936, 1.6751611
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8057923, 1.7871537
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1262112, 1.1298922
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7032866, 1.7107728
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1470642, 2.1188970
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2123876, 1.2107770
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6377254, 1.6255383

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 424

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7022618, upper bound: 0.7197374
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7022618, upper bound: 0.7197374
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7546811, 1.7642713
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0600019, 2.0596235
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8522296, 1.8534636
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6769972, 1.6851079
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8954439, 1.9025106
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1346467, 1.1265603
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7186055, 1.7398102
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2946959, 2.2942543
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2283232, 1.2293222
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6370049, 1.6321211

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 415

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7549316, upper bound: 0.7313094
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7549316, upper bound: 0.7312908
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7650027, 1.7539499
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0614214, 2.0582037
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8535190, 1.8521740
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6774025, 1.6847026
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8990154, 1.8989387
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1302154, 1.1309915
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7217650, 1.7366507
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2981958, 2.2907543
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2284904, 1.2291551
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6363254, 1.6328011

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2629

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7361997, upper bound: 0.7476127
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7361997, upper bound: 0.7476127
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7672710, 1.7667809
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0542498, 2.0549784
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8530259, 1.8472741
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6777344, 1.6847827
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9169898, 1.9247928
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1375787, 1.1333336
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7124133, 1.7309370
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2836590, 2.2883434
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2348156, 1.2335509
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6445670, 1.6421590

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7594046, upper bound: 0.7502965
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7592671, upper bound: 0.7527436
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7672682, 1.7667837
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0518198, 2.0574086
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8530931, 1.8472071
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6770792, 1.6854379
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9197645, 1.9220181
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1370373, 1.1338750
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7123170, 1.7310331
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2840548, 2.2879472
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2325490, 1.2358174
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6467447, 1.6399813

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2853

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 725

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7663762, upper bound: 0.7582820
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7623031, upper bound: 0.7619879
time: 5.56 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7333378, upper bound: 0.7352690
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7302629, upper bound: 0.7379446
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7426788, upper bound: 0.7393686
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7439834, upper bound: 0.7380641
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7148552, upper bound: 0.7105689
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7095009, upper bound: 0.7155619
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7103206, upper bound: 0.7166058
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7156799, upper bound: 0.7112068
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6988046, upper bound: 0.6884089
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6877246, upper bound: 0.6992446
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7015695, upper bound: 0.6999673
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6993819, upper bound: 0.7021532
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6748391, upper bound: 0.6659567
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6748391, upper bound: 0.6659567
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6923341, upper bound: 0.7053561
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.6952175, upper bound: 0.7020444
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7603544, upper bound: 0.7527944
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7536113, upper bound: 0.7595194
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7594843, upper bound: 0.7548790
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7595130, upper bound: 0.7548785
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7506189, upper bound: 0.7539642
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7476512, upper bound: 0.7569611
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7022618, upper bound: 0.7197374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7022618, upper bound: 0.7197374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7549316, upper bound: 0.7313094
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7549316, upper bound: 0.7312908
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7361997, upper bound: 0.7476127
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7361997, upper bound: 0.7476127
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7594046, upper bound: 0.7502965
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7592671, upper bound: 0.7527436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7663762, upper bound: 0.7582820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 5, lower bound: -0.7623031, upper bound: 0.7619879

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7320571, 1.7143116
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0039392, 1.9996173
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7910500, 1.7942667
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6631064, 1.6762207
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9046783, 1.8973022
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1100314, 1.1093966
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6781611, 1.6631203
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2867794, 2.2842579
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2006440, 1.2126775
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6143661, 1.6040435

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7330555, upper bound: 0.7327797
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7308483, upper bound: 0.7349868
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7312775, 1.7159009
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0092120, 1.9948864
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7949595, 1.7873943
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6631870, 1.6761949
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9072132, 1.8947515
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1066093, 1.1127537
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6720929, 1.6730843
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2868004, 2.2842369
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2135870, 1.1980405
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6154218, 1.6037197

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2537

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7178282, upper bound: 0.7255129
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7178330, upper bound: 0.7255084
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7361555, 1.7144492
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0608668, 2.0498385
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8519869, 1.8510156
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6519184, 1.6661661
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8436184, 1.8436520
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1417639, 1.1426827
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7323155, 1.7326984
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3029547, 2.3088636
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2049158, 1.1996953
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5951505, 1.6060877

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2629

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7333776, upper bound: 0.7300637
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7333776, upper bound: 0.7300637
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7300272, 1.7205777
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0642042, 2.0465019
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8512368, 1.8517656
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6530027, 1.6650820
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8554564, 1.8318141
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1416864, 1.1427603
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7273855, 1.7376285
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3066845, 2.3051333
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2001822, 1.2044287
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6179781, 1.5832620

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7437011, upper bound: 0.7355771
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7414939, upper bound: 0.7377819
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7630291, 1.7491705
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0470080, 2.0464246
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8516340, 1.8515692
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6598339, 1.6709461
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9060726, 1.8959560
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1282904, 1.1266838
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7386379, 1.7277751
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3059158, 2.2990818
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2296839, 1.2291354
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6154685, 1.6117926

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 425

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7134850, upper bound: 0.7102410
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7145247, upper bound: 0.7091978
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7624865, 1.7497451
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0642028, 2.0292692
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8507376, 1.8528252
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6871233, 1.6438506
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9043980, 1.8976364
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1256721, 1.1294862
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7394695, 1.7275028
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3043480, 2.3006477
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2314324, 1.2275548
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6148467, 1.6124239

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7003278, upper bound: 0.7060913
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7021103, upper bound: 0.7020637
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7602592, 1.7415447
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0651197, 2.0496428
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8497720, 1.8466055
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6696353, 1.6615210
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9064384, 1.9036667
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1277810, 1.1305695
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7217498, 1.6950941
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2832546, 2.2857614
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2235091, 1.2192993
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6136084, 1.6171150

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 415

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6961775, upper bound: 0.7034256
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6961775, upper bound: 0.7034256
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7595658, 1.7422390
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0674605, 2.0472128
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8479905, 1.8482585
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6695457, 1.6615613
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9073772, 1.9027123
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1286761, 1.1296735
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7148614, 1.7024915
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2808943, 2.2881222
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2224162, 1.2203231
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6142278, 1.6164932

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 415

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 410

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7151665, upper bound: 0.7095211
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7101492, upper bound: 0.7105549
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7127252, 1.7406828
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0342069, 2.0503762
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8590550, 1.8498068
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6455135, 1.6521747
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8962736, 1.8931227
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1127242, 1.1190743
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6681643, 1.7203538
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2906160, 2.2640462
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1498342, 1.2025616
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6191435, 1.5742373

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 415

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1983

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6955711, upper bound: 0.6799410
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6903118, upper bound: 0.6851563
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7156320, 1.7376850
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0377665, 2.0471399
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8614249, 1.8470449
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6472778, 1.6502593
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8969598, 1.8924365
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1109849, 1.1209261
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6717534, 1.7183075
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2935228, 2.2611399
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1487613, 1.2036349
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6206245, 1.5727701

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2629

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6779445, upper bound: 0.6897382
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6779445, upper bound: 0.6897384
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7321978, 1.7401986
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0399113, 2.0430000
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8563604, 1.8587568
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6473837, 1.6756153
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9004712, 1.9037290
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1183066, 1.1186247
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7276111, 1.6793342
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2772646, 2.2894549
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2167451, 1.1549102
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5932307, 1.6175141

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6994134, upper bound: 0.6997462
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7013298, upper bound: 0.6973074
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7321949, 1.7402015
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0374808, 2.0454302
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8564277, 1.8586895
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6467285, 1.6762705
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9032454, 1.9009542
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1177652, 1.1191661
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7275152, 1.6794302
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2776613, 2.2890587
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2144785, 1.1571767
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5954084, 1.6153364

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 158

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6902582, upper bound: 0.6904486
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6872625, upper bound: 0.6931733
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7476892, 1.7729890
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0533605, 2.0625141
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8639374, 1.8616006
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6812520, 1.6904478
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9111342, 1.9051106
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1313952, 1.1351703
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7428427, 1.7460504
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3075514, 2.2987881
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2387366, 1.2320874
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6360316, 1.6206717

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 961

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6732223, upper bound: 0.6599036
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6687945, upper bound: 0.6643394
time: 8.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7483130, 1.7688761
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0536890, 2.0626750
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8648338, 1.8599205
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6821833, 1.6999378
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9097152, 1.9084029
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1314832, 1.1361395
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7448826, 1.7400174
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3005342, 2.3019042
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2407346, 1.2330284
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6327367, 1.6233187

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2530

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 772

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 677

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6560544, upper bound: 0.6584258
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6673071, upper bound: 0.6471695
time: 5.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7367206, 1.7566452
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0498567, 2.0575922
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8656130, 1.8565872
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6809845, 1.6999815
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8911185, 1.8865423
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1303616, 1.1366668
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7368574, 1.7324526
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2889643, 2.2851739
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2325628, 1.2303872
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6354547, 1.6233311

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1983

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6679709, upper bound: 0.6791496
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6661098, upper bound: 0.6810179
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7354469, 1.7577910
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0488133, 2.0587022
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8623118, 1.8598883
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6813297, 1.6996834
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8916755, 1.8846941
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1314363, 1.1356478
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7347913, 1.7347934
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2902346, 2.2839050
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2354777, 1.2274728
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6354766, 1.6233087

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 424

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6791476, upper bound: 0.6858728
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6791476, upper bound: 0.6858728
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7582340, 1.7441244
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9794722, 2.0147371
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8059793, 1.7968414
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6365600, 1.6747231
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7596416, 1.7822275
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1299367, 1.1223488
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6223822, 1.6107426
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1078463, 2.1297364
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1962507, 1.1894896
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5523319, 1.5633459

Time for backsubstitution: 4.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 158

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7525926, upper bound: 0.7406657
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7496316, upper bound: 0.7436744
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7576594, 1.7446990
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9966278, 1.9975820
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8050828, 1.7977376
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6636553, 1.6476274
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7579665, 1.7839026
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1271343, 1.1251512
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6226540, 1.6104705
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1062803, 2.1313024
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1979992, 1.1877412
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5517101, 1.5639682

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7183725, upper bound: 0.7247415
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7183725, upper bound: 0.7247414
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7541227, 1.7501760
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0061302, 2.0141311
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8118672, 1.8079648
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6714344, 1.6778607
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7637143, 1.7903647
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1299453, 1.1276860
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6075854, 1.6272826
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1070671, 2.1288853
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1927140, 1.1983771
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5591617, 1.5584908

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7587954, upper bound: 0.7423748
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7435054, upper bound: 0.7541393
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7555218, 1.7503870
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0065207, 2.0143230
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8134236, 1.8103216
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6715002, 1.6777050
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7649221, 1.7888470
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1320910, 1.1247022
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6082072, 1.6286390
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1071281, 2.1338139
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1929464, 1.1984506
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5624881, 1.5572262

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7549823, upper bound: 0.7506302
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7552629, upper bound: 0.7503499
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7175360, 1.7238669
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9858875, 1.9908941
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7751546, 1.7785821
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6672363, 1.6710093
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7933769, 1.7861807
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1108074, 1.1062137
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6762362, 1.6653736
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1459446, 2.1187220
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1891618, 1.2036200
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6291246, 1.6217532

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 963

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7483924, upper bound: 0.7520018
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7486396, upper bound: 0.7513445
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7167559, 1.7255859
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9904728, 1.9861631
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7790637, 1.7721515
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6672621, 1.6709833
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7959113, 1.7836304
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1073852, 1.1097969
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6671004, 1.6753376
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1459656, 2.1187010
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2021048, 1.1889675
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6301799, 1.6214142

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7353093, upper bound: 0.7447847
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7354888, upper bound: 0.7446057
time: 3.77 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7330555, upper bound: 0.7327797
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7308483, upper bound: 0.7349868
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7178282, upper bound: 0.7255129
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7178330, upper bound: 0.7255084
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7333776, upper bound: 0.7300637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7333776, upper bound: 0.7300637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7437011, upper bound: 0.7355771
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7414939, upper bound: 0.7377819
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7134850, upper bound: 0.7102410
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7145247, upper bound: 0.7091978
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7003278, upper bound: 0.7060913
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7021103, upper bound: 0.7020637
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6961775, upper bound: 0.7034256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6961775, upper bound: 0.7034256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7151665, upper bound: 0.7095211
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7101492, upper bound: 0.7105549
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6955711, upper bound: 0.6799410
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6903118, upper bound: 0.6851563
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6779445, upper bound: 0.6897382
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6779445, upper bound: 0.6897384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6994134, upper bound: 0.6997462
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7013298, upper bound: 0.6973074
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6902582, upper bound: 0.6904486
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6872625, upper bound: 0.6931733
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6732223, upper bound: 0.6599036
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6687945, upper bound: 0.6643394
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6560544, upper bound: 0.6584258
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6673071, upper bound: 0.6471695
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6679709, upper bound: 0.6791496
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6661098, upper bound: 0.6810179
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6791476, upper bound: 0.6858728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.6791476, upper bound: 0.6858728
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7525926, upper bound: 0.7406657
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7496316, upper bound: 0.7436744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7183725, upper bound: 0.7247415
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7183725, upper bound: 0.7247414
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7587954, upper bound: 0.7423748
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7435054, upper bound: 0.7541393
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7549823, upper bound: 0.7506302
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7552629, upper bound: 0.7503499
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7483924, upper bound: 0.7520018
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7486396, upper bound: 0.7513445
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7353093, upper bound: 0.7447847
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.11
Output dim: 5, lower bound: -0.7354888, upper bound: 0.7446057
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7022618, upper bound: 0.7197374
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7022618, upper bound: 0.7197374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7549316, upper bound: 0.7313094
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7549316, upper bound: 0.7312908
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7361997, upper bound: 0.7476127
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7361997, upper bound: 0.7476127
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7594046, upper bound: 0.7502965
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7592671, upper bound: 0.7527436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7663762, upper bound: 0.7582820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.11
Output dim: 5, lower bound: -0.7623031, upper bound: 0.7619879
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.139000654220581
rel_dist={5: [-0.78768892317715, 0.7876895182364123]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2220

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 677

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6910992, upper bound: 0.7003447
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7003427, upper bound: 0.6911013
time: 4.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 5, lower bound: -0.6910992, upper bound: 0.7003447
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.89
Output dim: 5, lower bound: -0.7003427, upper bound: 0.6911013

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6294937, 1.6267724
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9371357, 1.9446898
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7012634, 1.7037899
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5631294, 1.5700736
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7070966, 1.6882510
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0312192, 1.0423266
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5999694, 1.6066370
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1885653, 2.1824913
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1087842, 1.1180975
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5488062, 1.5336480

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6908298, upper bound: 0.6971742
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6879185, upper bound: 0.7000692
time: 4.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6267724, 1.6294937
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9446898, 1.9371357
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7037897, 1.7012634
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5700736, 1.5631292
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6882510, 1.7070966
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0423267, 1.0312191
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6066370, 1.5999694
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1824913, 2.1885657
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1180975, 1.1087842
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5336475, 1.5488062

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1507

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6969041, upper bound: 0.6887556
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6979999, upper bound: 0.6876478
time: 3.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.93
Output dim: 5, lower bound: -0.6908298, upper bound: 0.6971742
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.93
Output dim: 5, lower bound: -0.6879185, upper bound: 0.7000692
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.93
Output dim: 5, lower bound: -0.6969041, upper bound: 0.6887556
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.93
Output dim: 5, lower bound: -0.6979999, upper bound: 0.6876478

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6296210, 1.6269560
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9371734, 1.9447503
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6995282, 1.7023649
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5631356, 1.5700920
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7066545, 1.6878924
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0313241, 1.0424109
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5998464, 1.6066294
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1889133, 2.1829019
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1086316, 1.1179781
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5487680, 1.5335741

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6892620, upper bound: 0.6908061
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6861762, upper bound: 0.6955333
time: 5.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6296773, 1.6268995
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9371963, 1.9447274
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6998382, 1.7020547
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5631480, 1.5700798
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7067380, 1.6878090
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0313036, 1.0424316
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5999618, 1.6065142
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1889763, 2.1828394
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1086650, 1.1179447
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5487328, 1.5336099

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6470948, upper bound: 0.6594707
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6470948, upper bound: 0.6594707
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6035814, 1.6013994
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9467182, 1.9418335
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7077439, 1.7046168
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5576568, 1.5515797
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6183906, 1.6467063
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0553186, 1.0441489
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6322227, 1.6216109
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2003708, 2.2094297
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0999844, 1.0868845
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5073647, 1.5407858

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 976

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6949261, upper bound: 0.6790913
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6872555, upper bound: 0.6867818
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5986786, 1.6063023
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9493871, 1.9391642
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7071435, 1.7052169
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5585237, 1.5507123
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6278610, 1.6372361
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0552564, 1.0442110
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6282783, 1.6255548
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2033558, 2.2064462
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0961978, 1.0906713
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5256276, 1.5225234

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 192

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6927692, upper bound: 0.6831427
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6934951, upper bound: 0.6822919
time: 4.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6892620, upper bound: 0.6908061
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6861762, upper bound: 0.6955333
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6470948, upper bound: 0.6594707
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6470948, upper bound: 0.6594707
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6949261, upper bound: 0.6790913
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6872555, upper bound: 0.6867818
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6927692, upper bound: 0.6831427
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.02
Output dim: 5, lower bound: -0.6934951, upper bound: 0.6822919

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6283555, 1.6262188
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9296494, 1.9374478
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6924925, 1.6990290
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5564322, 1.5610628
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6528854, 1.6377604
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0299407, 1.0390866
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5891256, 1.5924253
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1830387, 2.1773963
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0970824, 1.1070062
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5484409, 1.5351052

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1836

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6852072, upper bound: 0.6757598
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6740933, upper bound: 0.6867513
time: 3.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6281891, 1.6256907
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9298716, 1.9372258
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6953292, 1.6953294
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5541067, 1.5632281
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6585360, 1.6341236
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0279998, 1.0410422
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5856423, 1.5949602
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1834078, 2.1770267
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0976593, 1.1071670
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5507669, 1.5332470

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1948

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6828228, upper bound: 0.6870008
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6763430, upper bound: 0.6920536
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6296182, 1.6307254
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9402599, 1.9443727
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7103469, 1.7014027
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6087732, 1.5673325
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7058444, 1.7077379
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0309958, 1.0441160
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5997977, 1.6064529
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2049408, 2.1823506
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1080196, 1.1161975
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5486999, 1.5386400

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1983

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345944
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345923
time: 5.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6296773, 1.6268404
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9368410, 1.9447274
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6991861, 1.7020547
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5604000, 1.5700798
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7067380, 1.6869154
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0313036, 1.0421239
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5999618, 1.6063502
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1884880, 2.1828394
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1086650, 1.1172994
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5487328, 1.5335774

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 410

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345944
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345924
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6035814, 1.6013949
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9466839, 1.9418175
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7057652, 1.7033219
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5574164, 1.5513525
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6143618, 1.6434054
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0549958, 1.0436094
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6318040, 1.6209979
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1947803, 2.2051649
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0991662, 1.0858190
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5072322, 1.5406857

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 186

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6866081, upper bound: 0.6709125
time: 7.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6865943, upper bound: 0.6718536
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6035767, 1.6013997
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9467025, 1.9417992
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7064490, 1.7026386
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5574298, 1.5513391
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6150894, 1.6426778
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0547788, 1.0438261
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6316094, 1.6211925
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1961060, 2.2038383
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0989189, 1.0860662
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5072651, 1.5406532

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1397

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6817190, upper bound: 0.6815147
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6820237, upper bound: 0.6812121
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5902767, 1.6004765
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9470716, 1.9364235
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7005124, 1.6994984
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5526457, 1.5459242
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6184969, 1.6278515
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0425735, 1.0327013
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6249261, 1.6198394
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2030725, 2.2061987
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0968709, 1.0912473
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5207148, 1.5180798

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6922305, upper bound: 0.6742160
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6814358, upper bound: 0.6825828
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5924172, 1.5979004
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9466467, 1.9368484
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7013502, 1.6985855
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5537348, 1.5448346
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6184759, 1.6278713
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0437465, 1.0315261
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6225629, 1.6221504
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2031078, 2.2061629
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0967739, 1.0913444
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5211830, 1.5176110

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6909914, upper bound: 0.6713332
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6826918, upper bound: 0.6796906
time: 4.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6852072, upper bound: 0.6757598
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6740933, upper bound: 0.6867513
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6828228, upper bound: 0.6870008
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6763430, upper bound: 0.6920536
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345944
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345923
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345944
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6224314, upper bound: 0.6345924
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6866081, upper bound: 0.6709125
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6865943, upper bound: 0.6718536
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6817190, upper bound: 0.6815147
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6820237, upper bound: 0.6812121
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6922305, upper bound: 0.6742160
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6814358, upper bound: 0.6825828
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6909914, upper bound: 0.6713332
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.03
Output dim: 5, lower bound: -0.6826918, upper bound: 0.6796906

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5316906, 1.5589609
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9000335, 1.9100220
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6908493, 1.7008879
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5572672, 1.5635297
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6383891, 1.6165671
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0285139, 1.0296692
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4715588, 1.4998960
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1818819, 2.1763186
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0955210, 1.1054473
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5478001, 1.5335021

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6837413, upper bound: 0.6754674
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6849299, upper bound: 0.6747161
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5610981, 1.5295537
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9022231, 1.9078324
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6943512, 1.6973863
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5588989, 1.5618980
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6316924, 1.6232636
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0205235, 1.0376596
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4965966, 1.4748585
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1819620, 2.1762390
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0955236, 1.1054447
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5468383, 1.5344639

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2537

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6633298, upper bound: 0.6756219
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6633413, upper bound: 0.6756120
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6133122, 1.6047776
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9092913, 1.9216270
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6740017, 1.6751482
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5486889, 1.5586460
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5713153, 1.5628171
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0236953, 1.0342313
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5791674, 1.5889192
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.0188684, 2.0327744
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0912871, 1.1005958
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5355229, 1.5233865

Time for backsubstitution: 4.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6822943, upper bound: 0.6789969
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6719050, upper bound: 0.6864266
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6072764, 1.6108141
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9142723, 1.9166460
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6751480, 1.6740015
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5495243, 1.5578103
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5872326, 1.5469031
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0211889, 1.0367379
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5796013, 1.5884852
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.0389891, 2.0124869
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0910885, 1.1007947
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5409069, 1.5180035

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6742405, upper bound: 0.6917819
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6760659, upper bound: 0.6899993
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6024981, 1.6009972
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.7479978, 1.7592163
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6792696, 1.6834576
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5163965, 1.5110261
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5643396, 1.5847957
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0074575, 1.0016460
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6317825, 1.6209888
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1624870, 2.1710367
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0410993, 1.0421020
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5042152, 1.5362101

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2220

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6860736, upper bound: 0.6708621
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6865555, upper bound: 0.6703774
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6031842, 1.6009402
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.7634668, 1.7431312
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6921680, 1.6768262
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5170898, 1.5148590
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5586858, 1.5933833
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0130322, 1.0014253
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6317954, 1.6209769
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1638889, 2.1728706
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0560181, 1.0277522
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5029020, 1.5376687

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1402

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6854417, upper bound: 0.6701659
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6854556, upper bound: 0.6704687
time: 5.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6024084, 1.6000631
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9461827, 1.9415915
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7049365, 1.6992407
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5573573, 1.5513916
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6138163, 1.6426184
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0525784, 1.0440120
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6295633, 1.6196587
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1916351, 2.1994162
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0986853, 1.0856937
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5039396, 1.5383396

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2375

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6564972, upper bound: 0.6566501
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6564972, upper bound: 0.6566501
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6035767, 1.6002319
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9464946, 1.9417992
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7064490, 1.7011261
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5574298, 1.5512671
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6150894, 1.6414044
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0547788, 1.0416257
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6300759, 1.6211925
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1916847, 2.2038383
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0989189, 1.0858326
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5072651, 1.5373282

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6711250, upper bound: 0.6735638
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6743740, upper bound: 0.6701180
time: 5.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5891695, 1.6000993
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9465914, 1.9380980
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7020428, 1.7060580
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5524664, 1.5458517
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6179738, 1.6271269
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0368824, 1.0216517
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6141219, 1.6119246
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1941366, 2.2002916
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0961931, 1.0915513
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5189800, 1.5108776

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6895615, upper bound: 0.6715425
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6900534, upper bound: 0.6711398
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5898991, 1.5993700
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9487457, 1.9356923
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7070725, 1.7010291
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5525732, 1.5457144
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6177726, 1.6272621
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0315239, 1.0258734
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6170111, 1.6090362
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1967382, 2.1972623
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0971751, 1.0905693
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5140023, 1.5163450

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6811487, upper bound: 0.6823462
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6767005, upper bound: 0.6823505
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5838599, 1.5916684
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9364138, 1.9292040
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6938758, 1.6930070
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5410910, 1.5337226
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6174989, 1.6274431
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0411344, 1.0275223
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6117172, 1.6129417
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1934032, 2.1987834
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0925555, 1.0862677
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5168357, 1.5144501

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2216

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2930

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6818333, upper bound: 0.6656594
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6817722, upper bound: 0.6656613
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5861850, 1.5893428
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9390025, 1.9266148
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6957717, 1.6911111
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5426235, 1.5321901
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6180482, 1.6268940
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0397429, 1.0289137
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6133542, 1.6113048
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1957283, 2.1964583
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0916972, 1.0871260
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5180216, 1.5132642

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1402

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6813172, upper bound: 0.6785303
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6809019, upper bound: 0.6785181
time: 4.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6837413, upper bound: 0.6754674
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6849299, upper bound: 0.6747161
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6633298, upper bound: 0.6756219
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6633413, upper bound: 0.6756120
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6822943, upper bound: 0.6789969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6719050, upper bound: 0.6864266
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6742405, upper bound: 0.6917819
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6760659, upper bound: 0.6899993
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6860736, upper bound: 0.6708621
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6865555, upper bound: 0.6703774
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6854417, upper bound: 0.6701659
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6854556, upper bound: 0.6704687
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6564972, upper bound: 0.6566501
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6564972, upper bound: 0.6566501
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6711250, upper bound: 0.6735638
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6743740, upper bound: 0.6701180
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6895615, upper bound: 0.6715425
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6900534, upper bound: 0.6711398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6811487, upper bound: 0.6823462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6767005, upper bound: 0.6823505
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6818333, upper bound: 0.6656594
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6817722, upper bound: 0.6656613
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6813172, upper bound: 0.6785303
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.02
Output dim: 5, lower bound: -0.6809019, upper bound: 0.6785181

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5317307, 1.5589356
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8708949, 1.8752284
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6557727, 1.6568205
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5436854, 1.5525191
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6272163, 1.6054897
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0257566, 1.0274112
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4692411, 1.4977584
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1640444, 2.1613193
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0947423, 1.1039610
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5528212, 1.5386267

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6705419, upper bound: 0.6636476
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6705419, upper bound: 0.6636476
time: 4.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5316648, 1.5589767
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8652406, 1.8799407
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6467824, 1.6644192
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5463300, 1.5499475
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6273117, 1.6049819
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0260797, 1.0269119
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4689169, 1.4975786
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1668825, 2.1592174
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0942483, 1.1046686
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5529242, 1.5382991

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6412147, upper bound: 0.6382814
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6412147, upper bound: 0.6382814
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5610700, 1.5292511
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9046259, 1.9069164
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6886806, 1.6925063
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5578442, 1.5613346
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6334009, 1.6225293
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0193970, 1.0378897
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4899642, 1.4419334
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1840343, 2.1750379
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0844040, 1.1021103
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5406828, 1.5283833

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 186

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1446

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6526602, upper bound: 0.6732227
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6608896, upper bound: 0.6653494
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5610981, 1.5295260
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9013071, 1.9078324
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6894712, 1.6973863
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5583353, 1.5618980
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6309581, 1.6232636
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0205235, 1.0365331
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4965966, 1.4682257
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1807594, 2.1762390
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0955236, 1.0943251
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5407577, 1.5344639

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 429

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 424

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6485576, upper bound: 0.6637194
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6485576, upper bound: 0.6637194
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6121559, 1.6043501
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9072452, 1.9219868
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6755321, 1.6817091
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5484452, 1.5585399
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5705590, 1.5619252
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0242593, 1.0296575
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5725489, 1.5851889
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.0197506, 2.0376911
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0911565, 1.1014469
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5388246, 1.5208945

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6764955, upper bound: 0.6709090
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6742188, upper bound: 0.6732019
time: 6.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6128850, 1.6036208
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9096513, 1.9195809
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6805613, 1.6766794
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5485826, 1.5584025
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5704236, 1.5620077
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0191216, 1.0350156
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5754366, 1.5822997
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.0227795, 2.0336571
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0921383, 1.1004648
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5333571, 1.5266881

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6465993, upper bound: 0.6613176
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6465851, upper bound: 0.6613177
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6073160, 1.6107895
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8853788, 1.8818524
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6400743, 1.6299336
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5359421, 1.5468717
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5760603, 1.5358260
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0184320, 1.0343821
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5772839, 1.5862696
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.0211515, 1.9974871
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0903094, 1.0993080
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5458469, 1.5231266

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6685084, upper bound: 0.6865060
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6688128, upper bound: 0.6862245
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6072512, 1.6108303
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8794785, 1.8865616
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6310802, 1.6380315
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5385876, 1.5442283
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5761557, 1.5345075
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0185832, 1.0339812
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5773544, 1.5861681
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.0239897, 1.9953852
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0898159, 1.1000158
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5460300, 1.5222850

Time for backsubstitution: 4.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6503200, upper bound: 0.6659086
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6503056, upper bound: 0.6659093
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6029119, 1.6014874
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.7468371, 1.7577412
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6801555, 1.6840999
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5162034, 1.5105128
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5634675, 1.5842352
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0078712, 1.0021230
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6315346, 1.6214876
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1619787, 2.1705928
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0392971, 1.0398319
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5038791, 1.5359869

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1689

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6802830, upper bound: 0.6628104
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6780185, upper bound: 0.6650719
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6029873, 1.6014121
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.7465224, 1.7580562
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6799119, 1.6843433
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5158830, 1.5108335
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5637789, 1.5839238
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0079343, 1.0020598
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6322813, 1.6207411
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1620417, 2.1705289
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0388293, 1.0402997
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5039921, 1.5358739

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6556017, upper bound: 0.6396023
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6556017, upper bound: 0.6396023
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6066418, 1.6061108
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.7490914, 1.7279346
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6766491, 1.6633565
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5145178, 1.5127554
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5612397, 1.5943003
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0179363, 1.0066596
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6179895, 1.6083033
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1642828, 2.1732264
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0562599, 1.0280750
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5025382, 1.5370274

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6750966, upper bound: 0.6633312
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6784849, upper bound: 0.6598382
time: 7.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6083546, 1.6043999
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.7490447, 1.7287557
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6786985, 1.6613071
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5154753, 1.5122871
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5626855, 1.5959373
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0186492, 1.0063295
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6191216, 1.6073222
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1642447, 2.1732645
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0563409, 1.0279939
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5023384, 1.5373044

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6734596, upper bound: 0.6587363
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6737287, upper bound: 0.6584692
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6019135, 1.6027148
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9457541, 1.9414258
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7040486, 1.6999302
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5564270, 1.5428689
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6116590, 1.6393256
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0525069, 1.0431650
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6277518, 1.6226737
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1945066, 2.1966734
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0966051, 1.0828607
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5038824, 1.5356464

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6550994, upper bound: 0.6560921
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6559216, upper bound: 0.6556474
time: 3.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6024084, 1.5995681
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9460168, 1.9415915
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7049365, 1.6983526
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5573573, 1.5504608
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6105232, 1.6426184
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0525784, 1.0439403
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6295633, 1.6178477
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1888933, 2.1994162
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0986853, 1.0836135
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5012469, 1.5383396

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1397

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6546383, upper bound: 0.6548898
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6547361, upper bound: 0.6547944
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5927305, 1.5756235
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9154267, 1.9009323
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7041755, 1.6960473
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5507426, 1.5404136
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5911460, 1.6216364
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0478933, 1.0380917
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6187797, 1.6123226
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1808357, 2.1910563
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0683794, 1.0583488
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5024729, 1.5341740

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6593133, upper bound: 0.6619284
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6594931, upper bound: 0.6617491
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5789680, 1.5893862
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9056277, 1.9102552
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7013712, 1.6988518
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5465760, 1.5442224
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5953217, 1.6174583
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0508122, 1.0347402
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6212058, 1.6098974
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1789036, 2.1929884
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0714343, 1.0552933
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5041103, 1.5314474

Time for backsubstitution: 4.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6630201, upper bound: 0.6653580
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6696532, upper bound: 0.6586849
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5891581, 1.6001914
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9465117, 1.9380403
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7020037, 1.7060468
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5524540, 1.5458143
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6175508, 1.6271007
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0368309, 1.0216057
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6140795, 1.6119790
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1939793, 2.2002397
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0961530, 1.0915163
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5185804, 1.5106244

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512259, upper bound: 0.6325779
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6512259, upper bound: 0.6325779
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5891695, 1.6000872
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9465914, 1.9380188
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7020428, 1.7060192
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5524292, 1.5458517
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6179476, 1.6271269
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0368824, 1.0216002
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6141219, 1.6118824
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1940842, 2.2002916
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0961931, 1.0915115
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5187273, 1.5108776

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 424

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6752652, upper bound: 0.6551970
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6742158, upper bound: 0.6562466
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5880938, 1.5975626
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9392681, 1.9242706
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7091916, 1.7036991
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5471225, 1.5397394
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5957065, 1.6074157
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0289606, 1.0228770
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6175175, 1.6095076
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1850033, 2.1858449
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1023977, 1.0939790
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5088105, 1.5128951

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 425

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6779666, upper bound: 0.6821961
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6810006, upper bound: 0.6785068
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5880919, 1.5964773
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9373240, 1.9260752
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7092450, 1.7031479
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5461760, 1.5402634
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.5979261, 1.6056366
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0286402, 1.0233101
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6174822, 1.6095846
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1853199, 2.1835685
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1002524, 1.0957923
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5105529, 1.5120139

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6665158, upper bound: 0.6753293
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6700147, upper bound: 0.6718813
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5831437, 1.5904722
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9380512, 1.9278908
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6909857, 1.6928635
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5410032, 1.5335698
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6156015, 1.6252232
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0410876, 1.0274780
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6120248, 1.6128006
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1965399, 2.1968694
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0924807, 1.0860204
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5167160, 1.5149899

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1446

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6712895, upper bound: 0.6632633
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6794275, upper bound: 0.6535257
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5838599, 1.5909522
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9351001, 1.9292040
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6938758, 1.6901169
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5410910, 1.5336349
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6152792, 1.6274431
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0411344, 1.0274758
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6115761, 1.6129417
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1914892, 2.1987834
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0925555, 1.0861926
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5168357, 1.5143299

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1829

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6410346, upper bound: 0.6247445
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6410346, upper bound: 0.6247445
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5896444, 1.5945122
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9246273, 1.9121926
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6802557, 1.6776428
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5400515, 1.5305758
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6206007, 1.6308928
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0446491, 1.0345333
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5996995, 1.5986309
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1961222, 2.1968136
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0919397, 1.0874498
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5176578, 1.5127001

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6755007, upper bound: 0.6738123
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6760663, upper bound: 0.6734943
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5913544, 1.5927999
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9238062, 1.9122393
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6823037, 1.6755936
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5405202, 1.5296183
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6189637, 1.6294465
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0449793, 1.0338202
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6006804, 1.5974972
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1960840, 2.1968513
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0920210, 1.0873687
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5173807, 1.5128999

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 961

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6676387, upper bound: 0.6657101
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6676387, upper bound: 0.6657101
time: 4.25 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6705419, upper bound: 0.6636476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6705419, upper bound: 0.6636476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6412147, upper bound: 0.6382814
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6412147, upper bound: 0.6382814
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6526602, upper bound: 0.6732227
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6608896, upper bound: 0.6653494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6485576, upper bound: 0.6637194
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6485576, upper bound: 0.6637194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6764955, upper bound: 0.6709090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6742188, upper bound: 0.6732019
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6465993, upper bound: 0.6613176
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6465851, upper bound: 0.6613177
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6685084, upper bound: 0.6865060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6688128, upper bound: 0.6862245
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6503200, upper bound: 0.6659086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6503056, upper bound: 0.6659093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6802830, upper bound: 0.6628104
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6780185, upper bound: 0.6650719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6556017, upper bound: 0.6396023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6556017, upper bound: 0.6396023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6750966, upper bound: 0.6633312
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6784849, upper bound: 0.6598382
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6734596, upper bound: 0.6587363
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6737287, upper bound: 0.6584692
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6550994, upper bound: 0.6560921
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6559216, upper bound: 0.6556474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6546383, upper bound: 0.6548898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6547361, upper bound: 0.6547944
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6593133, upper bound: 0.6619284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6594931, upper bound: 0.6617491
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6630201, upper bound: 0.6653580
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6696532, upper bound: 0.6586849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6512259, upper bound: 0.6325779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6512259, upper bound: 0.6325779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6752652, upper bound: 0.6551970
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6742158, upper bound: 0.6562466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6779666, upper bound: 0.6821961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6810006, upper bound: 0.6785068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6665158, upper bound: 0.6753293
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6700147, upper bound: 0.6718813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6712895, upper bound: 0.6632633
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6794275, upper bound: 0.6535257
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6410346, upper bound: 0.6247445
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6410346, upper bound: 0.6247445
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6755007, upper bound: 0.6738123
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6760663, upper bound: 0.6734943
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6676387, upper bound: 0.6657101
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.98
Output dim: 5, lower bound: -0.6676387, upper bound: 0.6657101

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5187197, 1.5317798
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8740058, 1.8629637
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6536698, 1.6552815
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5430622, 1.5426221
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6241384, 1.5981827
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0217388, 1.0234562
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4596646, 1.4899893
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1603479, 2.1583300
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0893378, 1.0968680
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5413232, 1.5219741

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2853

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6607137, upper bound: 0.6535943
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6607137, upper bound: 0.6535943
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5045748, 1.5589356
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8586307, 1.8752284
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6557727, 1.6547179
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5337887, 1.5525191
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6272163, 1.6024117
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0218015, 1.0274112
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4614718, 1.4977584
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1640444, 2.1576228
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0947423, 1.0985565
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5528212, 1.5271287

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1507

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6458323, upper bound: 0.6390389
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6481047, upper bound: 0.6363436
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.5516698, 1.5156732
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8970504, 1.8967803
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.6710730, 1.6627553
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5568838, 1.5615222
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.6189146, 1.6013274
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0007635, 1.0244315
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.4568162, 1.4195163
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1670952, 2.1540031
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.0835271, 1.1009481
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5397973, 1.5241690

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 976
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 963

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1412

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6523907, upper bound: 0.6689984
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6524280, upper bound: 0.6729211
time: 3.92 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 12.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.81
Output dim: 5, lower bound: -0.6607137, upper bound: 0.6535943
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.81
Output dim: 5, lower bound: -0.6607137, upper bound: 0.6535943
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.81
Output dim: 5, lower bound: -0.6458323, upper bound: 0.6390389
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.81
Output dim: 5, lower bound: -0.6481047, upper bound: 0.6363436
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.81
Output dim: 5, lower bound: -0.6523907, upper bound: 0.6689984
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.81
Output dim: 5, lower bound: -0.6524280, upper bound: 0.6729211
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6608896, upper bound: 0.6653494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6485576, upper bound: 0.6637194
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6485576, upper bound: 0.6637194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6764955, upper bound: 0.6709090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6742188, upper bound: 0.6732019
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6465993, upper bound: 0.6613176
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6465851, upper bound: 0.6613177
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6685084, upper bound: 0.6865060
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6688128, upper bound: 0.6862245
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6503200, upper bound: 0.6659086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6503056, upper bound: 0.6659093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6802830, upper bound: 0.6628104
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6780185, upper bound: 0.6650719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6556017, upper bound: 0.6396023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6556017, upper bound: 0.6396023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6750966, upper bound: 0.6633312
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6784849, upper bound: 0.6598382
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6734596, upper bound: 0.6587363
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6737287, upper bound: 0.6584692
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6550994, upper bound: 0.6560921
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6559216, upper bound: 0.6556474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6546383, upper bound: 0.6548898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6547361, upper bound: 0.6547944
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6593133, upper bound: 0.6619284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6594931, upper bound: 0.6617491
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6630201, upper bound: 0.6653580
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6696532, upper bound: 0.6586849
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6512259, upper bound: 0.6325779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6512259, upper bound: 0.6325779
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6752652, upper bound: 0.6551970
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6742158, upper bound: 0.6562466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6779666, upper bound: 0.6821961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6810006, upper bound: 0.6785068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6665158, upper bound: 0.6753293
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6700147, upper bound: 0.6718813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6712895, upper bound: 0.6632633
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6794275, upper bound: 0.6535257
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6755007, upper bound: 0.6738123
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6760663, upper bound: 0.6734943
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6676387, upper bound: 0.6657101
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.81
Output dim: 5, lower bound: -0.6676387, upper bound: 0.6657101
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.0726536512374878
rel_dist={5: [-0.7059536409330516, 0.7059557061852892]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2405.50 seconds
