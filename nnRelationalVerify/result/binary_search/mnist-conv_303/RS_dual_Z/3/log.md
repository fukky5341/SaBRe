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
execution time: IAR + LP analysis = 13.14 + 32.08 = 45.22 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.78 seconds, max iter: 100)

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
Binary search time: 146.25 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_dual_Z) starts
Time budget: 3408.53 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2216

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.0000987, upper bound: 1.0001304
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.0001284, upper bound: 1.0001006
time: 3.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 5, lower bound: -1.0000987, upper bound: 1.0001304
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 5, lower bound: -1.0001284, upper bound: 1.0001006

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1311212, 2.1310632
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4729548, 2.4688177
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1593223, 2.1550262
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9637117, 1.9777381
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3216877, 2.3180985
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3352281, 1.3362260
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0845270, 2.0671380
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6800108, 2.6800108
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4557099, 1.4560250
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9668097, upper bound: 0.9676586
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9668097, upper bound: 0.9676585
time: 3.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1316342, 2.1311212
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4688177, 2.4731612
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1630950, 2.1593223
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9777384, 1.9837577
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3180985, 2.3219538
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3380417, 1.3352280
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0671377, 2.0788450
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6800108, 2.6800241
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4558139, 1.4557101
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9676567, upper bound: 0.9668117
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9676567, upper bound: 0.9668116
time: 3.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 11.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.03
Output dim: 5, lower bound: -0.9668097, upper bound: 0.9676586
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.03
Output dim: 5, lower bound: -0.9668097, upper bound: 0.9676585
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.03
Output dim: 5, lower bound: -0.9676567, upper bound: 0.9668117
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.03
Output dim: 5, lower bound: -0.9676567, upper bound: 0.9668116

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1299357, 2.1288404
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4720182, 2.4694114
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1569428, 2.1610322
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9664078, 1.9764283
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3115711, 2.3112981
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3356843, 1.3360660
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0820794, 2.0664499
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6767664, 2.6758871
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4550924, 1.4540811
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9483286, upper bound: 0.9559944
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9551479, upper bound: 0.9491775
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1311212, 2.1298780
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4729548, 2.4678817
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1593223, 2.1526470
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9624014, 1.9777381
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3148875, 2.3180985
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3350680, 1.3362260
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0838389, 2.0671380
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6758871, 2.6800108
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4557099, 1.4554074
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9483286, upper bound: 0.9559944
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9551479, upper bound: 0.9491775
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1304488, 2.1289828
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4678817, 2.4737551
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1606960, 2.1653283
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9794750, 1.9825177
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3089070, 2.3148720
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3384979, 1.3350680
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0631733, 2.0780673
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6767664, 2.6758986
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4551964, 1.4537662
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9491757, upper bound: 0.9551496
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9559949, upper bound: 0.9483290
time: 3.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1316342, 2.1299357
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4688177, 2.4722254
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1630950, 2.1569433
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9764280, 1.9837577
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3112984, 2.3219538
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3378816, 1.3352280
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0664496, 2.0788450
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6758871, 2.6800241
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4558139, 1.4550924
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9491757, upper bound: 0.9551498
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9559949, upper bound: 0.9483290
time: 3.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.45 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9483286, upper bound: 0.9559944
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9551479, upper bound: 0.9491775
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9483286, upper bound: 0.9559944
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9551479, upper bound: 0.9491775
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9491757, upper bound: 0.9551496
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9559949, upper bound: 0.9483290
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9491757, upper bound: 0.9551498
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.45
Output dim: 5, lower bound: -0.9559949, upper bound: 0.9483290

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1259747, 2.1054254
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4902020, 2.4174585
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1562977, 2.1146195
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9666176, 1.8998549
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3026681, 2.3182912
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3353047, 1.3359973
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0804400, 2.0416298
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6787167, 2.6719232
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4526181, 1.4499291
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1065207, 2.1288404
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4200654, 2.4694114
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1105309, 2.1610322
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8898344, 1.9764283
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3115711, 2.3023956
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3356843, 1.3356862
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0572596, 2.0664499
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6728020, 2.6758871
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4550924, 1.4516068
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1271601, 2.1064630
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4911375, 2.4159288
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1586771, 2.1062343
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9626112, 1.9011650
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3059845, 2.3250918
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3346882, 1.3361576
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0821991, 2.0423188
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6778364, 2.6760464
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4532356, 1.4512553
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1077061, 2.1298780
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4210014, 2.4678817
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1129093, 2.1526470
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8858280, 1.9777381
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3148875, 2.3091960
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3350680, 1.3358464
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0590191, 2.0671380
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6719227, 2.6800108
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4557099, 1.4529330
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1264877, 2.1055677
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4860682, 2.4218011
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1600504, 2.1189156
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9796839, 1.9059441
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3000045, 2.3218653
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3381178, 1.3349996
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0615301, 2.0532479
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6787176, 2.6719351
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4527221, 1.4496143
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213130
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213132
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1070337, 2.1289828
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4159288, 2.4737551
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1142836, 2.1653283
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9029016, 1.9825177
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3089070, 2.3059695
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3384979, 1.3346884
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0383530, 2.0780673
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6728029, 2.6758986
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4551964, 1.4512918
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170284
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170283
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1276731, 2.1065207
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4870033, 2.4202714
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1624498, 2.1105304
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9766378, 1.9071844
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3023958, 2.3289471
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3375015, 1.3351595
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0648060, 2.0540254
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6778374, 2.6760597
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4533396, 1.4509405
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213130
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213132
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1082191, 2.1299357
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4168644, 2.4722254
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1166821, 2.1569433
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8998547, 1.9837577
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3112984, 2.3130512
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3378816, 1.3348486
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0416298, 2.0788450
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6719236, 2.6800241
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4558139, 1.4526180
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170284
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170284
time: 3.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9170265, upper bound: 0.9221585
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9213135, upper bound: 0.9178724
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213130
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213132
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170284
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170283
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213130
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9178705, upper bound: 0.9213132
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170284
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.45
Output dim: 5, lower bound: -0.9221575, upper bound: 0.9170284

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1264644, 2.0843172
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4805450, 2.4003031
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1542320, 2.1136801
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9633904, 1.8877809
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2992063, 2.3055685
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3289573, 1.3318064
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0677357, 2.0333323
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6750202, 2.6696405
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4472136, 1.4409816
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1048665, 2.1054254
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4730468, 2.4174585
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1562977, 2.1125534
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9545441, 1.8998549
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3026681, 2.3148293
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3311138, 1.3359973
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0721426, 2.0416298
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6787167, 2.6682262
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4526181, 1.4445246
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1137023, 2.1077824
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4320636, 2.4571445
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1084642, 2.1600931
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8948755, 1.9666150
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3085155, 2.2896352
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3293371, 1.3314953
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0453482, 2.0583994
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6691055, 2.6736050
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4496875, 1.4428256
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0854125, 2.1288404
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4029102, 2.4694114
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1105309, 2.1589663
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8777609, 1.9764283
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3115711, 2.2989335
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3314934, 1.3356862
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0489621, 2.0664499
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6728020, 2.6721907
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4550924, 1.4462023
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1276493, 2.0853548
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4814806, 2.3987734
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1566105, 2.1052949
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9593868, 1.8890898
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3025227, 2.3123689
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3283408, 1.3319663
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0694947, 2.0340199
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6741400, 2.6737642
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4478307, 1.4423077
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1060514, 2.1064630
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4739819, 2.4159288
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1586771, 2.1041684
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9505377, 1.9011650
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3059845, 2.3216298
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3304973, 1.3361576
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0739012, 2.0423188
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6778364, 2.6723499
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4532356, 1.4458508
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1148872, 2.1088200
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4329987, 2.4556148
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1108427, 2.1517079
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8908730, 1.9679239
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3118320, 2.2964356
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3287206, 1.3316551
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0471077, 2.0590889
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6682262, 2.6777287
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4503055, 1.4441518
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0865974, 2.1298780
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4038458, 2.4678817
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1129093, 2.1505811
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8737545, 1.9777381
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3148875, 2.3057342
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3308771, 1.3358464
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0507216, 2.0671380
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6719227, 2.6763139
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4557099, 1.4475285
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1269774, 2.0844595
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4764104, 2.4046466
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1579838, 2.1179764
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9764576, 1.8938696
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2965426, 2.3091424
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3317707, 1.3308085
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0488253, 2.0449493
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6750202, 2.6696529
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4473176, 1.4406668
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1053796, 2.1055677
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4689131, 2.4218011
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1600504, 2.1168497
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9676104, 1.9059441
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3000045, 2.3184032
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3339269, 1.3349996
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0532331, 2.0532479
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6787176, 2.6682386
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4527221, 1.4442098
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1142154, 2.1079247
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4279256, 2.4614887
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1122169, 2.1643891
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9079428, 1.9727042
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3058519, 2.2932091
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3321507, 1.3304973
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0264416, 2.0700178
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6691055, 2.6736164
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4497919, 1.4425107
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0859256, 2.1289828
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3987737, 2.4737551
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1142836, 2.1632624
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8908272, 1.9825177
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3089070, 2.3025074
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3343070, 1.3346884
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0300560, 2.0780673
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6728029, 2.6722021
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4551964, 1.4458873
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1281624, 2.0854125
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4773455, 2.4031169
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1603832, 2.1095912
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9734125, 1.8951099
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2989335, 2.3162241
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3311541, 1.3309684
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0521011, 2.0457270
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6741409, 2.6737776
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4479351, 1.4419929
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1065650, 2.1065207
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4698486, 2.4202714
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1624498, 2.1084645
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9645634, 1.9071844
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3023958, 2.3254852
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3333104, 1.3351595
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0565085, 2.0540254
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6778374, 2.6723633
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4533396, 1.4455360
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
time: 3.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1154008, 2.1088777
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4288607, 2.4599590
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1146164, 2.1560040
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9048996, 1.9739444
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3082433, 2.3002911
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3315341, 1.3306572
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0297179, 2.0707960
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6682262, 2.6777415
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4504094, 1.4438369
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0871110, 2.1299357
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3997087, 2.4722254
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1166821, 2.1548772
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8877811, 1.9837577
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3112984, 2.3095894
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3336904, 1.3348486
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0333323, 2.0788450
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6719236, 2.6763268
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4558139, 1.4472135
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
time: 3.81 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601109, upper bound: 0.8610431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8601966, upper bound: 0.8609574
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8609555, upper bound: 0.8601985
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.01
Output dim: 5, lower bound: -0.8610412, upper bound: 0.8601128

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1715660, 2.0317035
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4705043, 2.3325005
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1226768, 2.0791349
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9609423, 1.8376477
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2765174, 2.3159904
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2934173, 1.3042883
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0696011, 2.0076189
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6381788, 2.6182423
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3885961, 1.4124072
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592468
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0738506, 2.0843172
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4127426, 2.4003031
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1196866, 2.1136801
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9132576, 1.8877809
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2992063, 2.2828796
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3014392, 1.3318064
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0420222, 2.0333323
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6236210, 2.6696405
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4472136, 1.3823645
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592496
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1481829, 2.0561213
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4495716, 2.3491809
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1247425, 2.0780082
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9537230, 1.8497703
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2802615, 2.3245168
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2988092, 1.3083589
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0724344, 2.0157864
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6418753, 2.6168280
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3938403, 1.4166491
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592468
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0522528, 2.1054254
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4052444, 2.4174585
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1217527, 2.1125534
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9044104, 1.8998549
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3026681, 2.2921405
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3035957, 1.3359973
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0464292, 2.0416298
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6273174, 2.6682262
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4526181, 1.3859074
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592496
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1614723, 2.0612752
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4305196, 2.4111216
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0893774, 2.1322827
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9186211, 1.9309218
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2904315, 2.3055856
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2937969, 1.3039771
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0612226, 2.0329278
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6376791, 2.6238728
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3924441, 1.4152131
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0610886, 2.1077824
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3642612, 2.4571445
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0739193, 2.1600931
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8447428, 1.9666150
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3085155, 2.2669463
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3018190, 1.3314953
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0196347, 2.0583994
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6177073, 2.6736050
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4496875, 1.3842084
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1407881, 2.0858603
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4061618, 2.4253969
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0914435, 2.1311560
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9056225, 1.9435213
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2942572, 2.3141499
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2991890, 1.3080478
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0609894, 2.0410955
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6413765, 2.6224580
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3972816, 1.4194543
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0327988, 2.1288404
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3351078, 2.4694114
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0759850, 2.1589663
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8276281, 1.9764283
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3115711, 2.2762446
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3039752, 1.3356862
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0232487, 2.0664499
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6214027, 2.6721907
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4550924, 1.3875852
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1727509, 2.0327411
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4714398, 2.3309708
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1250553, 2.0707498
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9569416, 1.8389571
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2798338, 2.3227909
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2928008, 1.3044482
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0713601, 2.0083077
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6372986, 2.6223664
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3892140, 1.4137340
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592468
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0750360, 2.0853548
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4136786, 2.3987734
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1220651, 2.1052949
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9092541, 1.8890898
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3025227, 2.2896800
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3008226, 1.3319663
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0437813, 2.0340199
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6227417, 2.6737642
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4478307, 1.3836906
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592496
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1493678, 2.0571589
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4505076, 2.3476512
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1271219, 2.0696232
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9497194, 1.8510795
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2835779, 2.3313174
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2981929, 1.3085190
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0741935, 2.0164754
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6409950, 2.6209517
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3944578, 1.4179759
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592468
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0534382, 2.1064630
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4061799, 2.4159288
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1241317, 2.1041684
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9004040, 1.9011650
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3059845, 2.2989411
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3029791, 1.3361576
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0481877, 2.0423188
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6264381, 2.6723499
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4532356, 1.3872337
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505647, upper bound: 0.8592496
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582086, upper bound: 0.8506637
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1626577, 2.0623131
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4314551, 2.4095919
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0917563, 2.1238976
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9146214, 1.9322312
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2937484, 2.3123860
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2931806, 1.3041371
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0629821, 2.0336163
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6367998, 2.6279955
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3930616, 1.4165399
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0622740, 2.1088200
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3651967, 2.4556148
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0762982, 2.1517079
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8407402, 1.9679239
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3118320, 2.2737470
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3012024, 1.3316551
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0213947, 2.0590889
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6168270, 2.6777287
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4503055, 1.3855345
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1419730, 2.0868979
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4070973, 2.4238672
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0938230, 2.1227708
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9016199, 1.9448307
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2975736, 2.3209503
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2985724, 1.3082079
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0627489, 2.0417843
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6404972, 2.6265812
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3978992, 1.4207811
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0339842, 2.1298780
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3360438, 2.4678817
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0783644, 2.1505811
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8236217, 1.9777381
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3148875, 2.2830453
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3033589, 1.3358464
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0250082, 2.0671380
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6205235, 2.6763139
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4557099, 1.3889112
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506504, upper bound: 0.8591639
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8582943, upper bound: 0.8505757
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1720791, 2.0318460
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4663682, 2.3368444
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1264291, 2.0834312
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9740095, 1.8437364
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2738538, 2.3195643
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2962304, 1.3032905
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0506902, 2.0192368
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6381788, 2.6182547
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3887000, 1.4120923
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0743637, 2.0844595
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4086080, 2.4046466
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1234393, 2.1179764
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9263239, 1.8938696
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2965426, 2.2864535
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3042525, 1.3308085
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0231118, 2.0449493
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6236219, 2.6696529
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4473176, 1.3820496
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1486959, 2.0562637
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4454365, 2.3535240
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1284957, 2.0823045
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9667892, 1.8558593
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2775979, 2.3280907
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3016225, 1.3073610
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0535245, 2.0274043
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6418762, 2.6168399
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3939443, 1.4163342
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0527658, 2.1055677
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4011102, 2.4218011
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1255054, 2.1168497
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9174776, 1.9059441
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3000045, 2.2957144
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3064088, 1.3349996
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0275197, 2.0532479
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6273184, 2.6682386
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4527221, 1.3855927
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1619854, 2.0614178
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4263806, 2.4154651
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0931301, 2.1365788
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9316883, 1.9370105
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2877679, 2.3091595
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2966105, 1.3029792
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0423150, 2.0445452
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6376801, 2.6238837
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3925481, 1.4148982
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506618, upper bound: 0.8582105
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8592477, upper bound: 0.8505640
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0616016, 2.1079247
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3601232, 2.4614887
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0776715, 2.1643891
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8578100, 1.9727042
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3058519, 2.2705202
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3046325, 1.3304973
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0007281, 2.0700178
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6177073, 2.6736164
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4497919, 1.3838935
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506618, upper bound: 0.8582105
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8592477, upper bound: 0.8505641
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1413012, 2.0860026
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4020238, 2.4297409
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0951967, 2.1354520
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9186897, 1.9496100
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2915936, 2.3177238
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3020025, 1.3070498
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0420828, 2.0527134
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6413765, 2.6224689
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3973856, 1.4191395
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506618, upper bound: 0.8582105
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8592477, upper bound: 0.8505640
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0333118, 2.1289828
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.3309712, 2.4737551
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0797381, 2.1632624
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.8406944, 1.9825177
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3089070, 2.2798185
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3067888, 1.3346884
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0043426, 2.0780673
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6214037, 2.6722021
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4551964, 1.3872702
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8506618, upper bound: 0.8582105
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8592477, upper bound: 0.8505641
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1732640, 2.0327990
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4673038, 2.3353148
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1288285, 2.0750461
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9709673, 1.8449762
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2762451, 2.3266461
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2956141, 1.3034503
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0539660, 2.0200148
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6372995, 2.6223774
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3893180, 1.4134190
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0755491, 2.0854125
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4095435, 2.4031169
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1258388, 2.1095912
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9232798, 1.8951099
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2989335, 2.2935352
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3036360, 1.3309684
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0263877, 2.0457270
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6227427, 2.6737776
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4479351, 1.3833756
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1498814, 2.0572169
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4463725, 2.3519943
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1308947, 2.0739193
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9637451, 1.8570983
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2799892, 2.3351727
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3010060, 1.3075211
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0568004, 2.0281825
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6409960, 2.6209631
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3945622, 1.4176610
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.0539513, 2.1065207
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4020457, 2.4202714
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.1279044, 2.1084645
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9144306, 1.9071844
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.3023958, 2.3027964
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.3057925, 1.3351595
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0307956, 2.0540254
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6264381, 2.6723633
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.4533396, 1.3869188
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8505761, upper bound: 0.8582962
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.8591620, upper bound: 0.8506523
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -2.1631708, 2.0623710
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.4273162, 2.4139354
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -2.0955296, 2.1281936
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.9286470, 1.9382513
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -2.2901587, 2.3162413
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.2959942, 1.3031392
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -2.0455918, 2.0453229
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.6368008, 2.6280088
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.3931661, 1.4162250
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.7069154, 1.7069154

Time for backsubstitution: 5.48 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.3380417823791504
rel_dist={5: [-1.0100506396824613, 1.010049981815225]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.29 seconds

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
time: 4.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.75
Output dim: 5, lower bound: -0.7787311, upper bound: 0.7793377
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.75
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

Time for backsubstitution: 5.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7633574, upper bound: 0.7635314
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7633574, upper bound: 0.7635315
time: 4.59 seconds

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

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7635318, upper bound: 0.7633575
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7635318, upper bound: 0.7633575
time: 4.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 5, lower bound: -0.7633574, upper bound: 0.7635314
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 5, lower bound: -0.7633574, upper bound: 0.7635315
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 5, lower bound: -0.7635318, upper bound: 0.7633575
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.24
Output dim: 5, lower bound: -0.7635318, upper bound: 0.7633575

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7683573, 1.7676723
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0652204, 2.0635910
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8646035, 1.8671589
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6935339, 1.6997964
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9081602, 1.9079897
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1364124, 1.1366508
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7518034, 1.7420347
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3059587, 2.3054099
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2410135, 1.2403812
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6319418, 1.6330667

Time for backsubstitution: 5.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7499807, upper bound: 0.7576550
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7574799, upper bound: 0.7501550
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7695422, 1.7683208
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0661564, 2.0626349
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8669825, 1.8619182
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6910295, 1.7011063
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9102330, 1.9147902
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1360271, 1.1368109
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7529030, 1.7427227
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3054094, 2.3095331
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2416310, 1.2412101
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6355152, 1.6316767

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7499807, upper bound: 0.7576550
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7574799, upper bound: 0.7501554
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7688699, 1.7677612
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0626349, 2.0679345
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8683562, 1.8698440
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.7017002, 1.7058859
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9064951, 1.9115636
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1392260, 1.1360270
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7399869, 1.7536521
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3059597, 2.3054214
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2411175, 1.2401844
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6355019, 1.6333303

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7501554, upper bound: 0.7574800
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7576545, upper bound: 0.7499827
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7700553, 1.7683570
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0635710, 2.0669785
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8707552, 1.8646033
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6997967, 1.7071259
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9079895, 1.9186454
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1388407, 1.1361872
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7420344, 1.7544298
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3054094, 2.3095465
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2417350, 1.2410133
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6392679, 1.6319418

Time for backsubstitution: 5.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7501554, upper bound: 0.7574800
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7576545, upper bound: 0.7499828
time: 4.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7499807, upper bound: 0.7576550
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7574799, upper bound: 0.7501550
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7499807, upper bound: 0.7576550
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7574799, upper bound: 0.7501554
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7501554, upper bound: 0.7574800
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7576545, upper bound: 0.7499827
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7501554, upper bound: 0.7574800
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.56
Output dim: 5, lower bound: -0.7576545, upper bound: 0.7499828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7571006, 1.7442572
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0571027, 2.0116382
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8467951, 1.8207462
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6649494, 1.6232231
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8992577, 1.9090219
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1360326, 1.1364655
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7414713, 1.7172146
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3056917, 2.3014455
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2385392, 1.2368584
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6067872, 1.6344347

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7449417, 1.7564158
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0132675, 2.0554752
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8181906, 1.8493502
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6169605, 1.6712122
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9091921, 1.8990870
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1362269, 1.1362711
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7269831, 1.7316999
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3019953, 2.3051424
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2374907, 1.2379069
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6333094, 1.6079121

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
time: 5.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7582855, 1.7449057
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0580378, 2.0106821
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8491745, 1.8155053
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6624460, 1.6245332
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9013305, 1.9158225
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1356473, 1.1366258
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7425704, 1.7179039
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3051414, 2.3055692
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2391562, 1.2376873
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6103601, 1.6330452

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7461271, 1.7570643
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0142031, 2.0545192
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8205700, 1.8441098
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6144562, 1.6725225
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9112654, 1.9058876
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1358418, 1.1364312
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7280827, 1.7323890
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3014450, 2.3092656
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2381077, 1.2387358
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6368828, 1.6065226

Time for backsubstitution: 5.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7576132, 1.7443461
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0545192, 2.0159807
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8505478, 1.8234313
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6731167, 1.6293123
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8975925, 1.9125962
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1388459, 1.1358418
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7296524, 1.7288327
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3056917, 2.3014579
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2386432, 1.2366617
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6103473, 1.6346984

Time for backsubstitution: 5.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7454548, 1.7565048
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0106821, 2.0598178
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8219433, 1.8520353
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6251268, 1.6773016
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9075274, 1.9026608
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1390402, 1.1356473
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7151670, 1.7433178
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3019953, 2.3051543
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2375951, 1.2377101
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6368694, 1.6081758

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310016
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310020
time: 6.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7587986, 1.7449419
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0554543, 2.0150247
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8529472, 1.8181906
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6712122, 1.6305525
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8990870, 1.9196777
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1384606, 1.1360021
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7316999, 1.7296104
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3051424, 2.3055820
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2392602, 1.2374905
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6141124, 1.6333094

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7466402, 1.7571006
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0116177, 2.0588617
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8243427, 1.8467951
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6232233, 1.6785417
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9090219, 1.9097428
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1386549, 1.1358075
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7172146, 1.7440956
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3014460, 2.3092785
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2382121, 1.2385390
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6406350, 1.6067872

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310016
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310019
time: 6.89 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7310018, upper bound: 0.7358136
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7355274, upper bound: 0.7313066
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310016
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310020
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7313067, upper bound: 0.7355295
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310016
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.71
Output dim: 5, lower bound: -0.7358125, upper bound: 0.7310019

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7494907, 1.7231491
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0446339, 1.9944825
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8447294, 1.8193843
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6584044, 1.6111491
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8957958, 1.8997717
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1304939, 1.1322746
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7304192, 1.7089171
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3019943, 2.2986326
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2331343, 1.2292395
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5951538, 1.6128531

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911618
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911610
time: 6.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7359920, 1.7442572
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0399470, 2.0116382
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8467951, 1.8186800
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6528759, 1.6232231
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8992577, 1.9055600
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1318417, 1.1364655
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7331734, 1.7172146
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3056917, 2.2977490
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2385392, 1.2314539
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6067872, 1.6228018

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911613
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911610
time: 6.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7415147, 1.7353077
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0143328, 2.0383198
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8161249, 1.8482769
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6155834, 1.6591384
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9057302, 1.8898134
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1306624, 1.1320801
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7164268, 1.7234025
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2982979, 2.3023310
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2320862, 1.2303920
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6216760, 1.5898347

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885153
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885145
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7238336, 1.7564158
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9961123, 2.0554752
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8181906, 1.8472841
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6048861, 1.6712122
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9091921, 1.8956251
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1320362, 1.1362711
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7186861, 1.7316999
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3019953, 2.3014455
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2374907, 1.2325025
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6333094, 1.5962791

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885153
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885145
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7506757, 1.7237976
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0455694, 1.9935265
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8471079, 1.8141437
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6559029, 1.6124580
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8978686, 1.9065723
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1301084, 1.1324344
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7315187, 1.7096047
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3014450, 2.3027563
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2337518, 1.2300684
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5987277, 1.6114635

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7371774, 1.7449057
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0408831, 2.0106821
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8491745, 1.8134396
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6503716, 1.6245332
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9013305, 1.9123607
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1314564, 1.1366258
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7342730, 1.7179039
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3051414, 2.3018723
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2391562, 1.2322829
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6103601, 1.6214118

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
time: 4.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7427001, 1.7359562
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0152683, 2.0373638
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8185034, 1.8430367
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6130819, 1.6604471
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9078035, 1.8966141
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1302769, 1.1322399
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7175269, 1.7240901
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2977486, 2.3064547
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2327037, 1.2312210
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6252499, 1.5884447

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885138
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885133
time: 5.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7250190, 1.7570643
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9970474, 2.0545192
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8205700, 1.8420441
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6023827, 1.6725225
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9112654, 1.9024258
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1316509, 1.1364312
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7197857, 1.7323890
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3014450, 2.3055687
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2381077, 1.2333313
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6368828, 1.5948892

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885138
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885134
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7500033, 1.7232380
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0420499, 1.9988260
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8484812, 1.8220694
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6665716, 1.6172378
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8941307, 1.9033456
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1333070, 1.1316509
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7186003, 1.7205343
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3019953, 2.2986450
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2332382, 1.2290428
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5987139, 1.6131172

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906034
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906037
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7365050, 1.7443461
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0373635, 2.0159807
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8505478, 1.8213654
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6610422, 1.6293123
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8975925, 1.9091339
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1346548, 1.1358418
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7213550, 1.7288327
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3056917, 2.2977610
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2386432, 1.2312572
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6103473, 1.6230655

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906035
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906036
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7420278, 1.7353966
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0117464, 2.0426633
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8198767, 1.8509622
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6237507, 1.6652272
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9040656, 1.8933873
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1334755, 1.1314564
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7046103, 1.7350194
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2982988, 2.3023434
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2321897, 1.2301953
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6252360, 1.5900984

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7243466, 1.7565048
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9935265, 2.0598178
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8219433, 1.8499694
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6130533, 1.6773016
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9075274, 1.8991990
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1348493, 1.1356473
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7068691, 1.7433178
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3019953, 2.3014574
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2375951, 1.2323056
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6368694, 1.5965424

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7511892, 1.7238338
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0429850, 1.9978700
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8508806, 1.8168290
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6646690, 1.6184781
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8956251, 1.9104276
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1329217, 1.1318107
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7206478, 1.7213120
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3014450, 2.3027697
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2338562, 1.2298715
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6024785, 1.6117277

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908021
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908023
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7376904, 1.7449419
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0382996, 2.0150247
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8529472, 1.8161247
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6591387, 1.6305525
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8990870, 1.9162159
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1342695, 1.1360021
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7234025, 1.7296104
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3051424, 2.3018856
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2392602, 1.2320861
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6141124, 1.6216764

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908021
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908023
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7432132, 1.7359924
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0126820, 2.0417073
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8222761, 1.8457220
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6218481, 1.6664674
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9055600, 1.9004693
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1330903, 1.1316162
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7066584, 1.7357972
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2977486, 2.3064675
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2328076, 1.2310240
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6290011, 1.5887098

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881560
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881561
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7255321, 1.7571006
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9944620, 2.0588617
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8243427, 1.8447292
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6111488, 1.6785417
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9090219, 1.9062810
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1344640, 1.1358075
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7089171, 1.7440956
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.3014460, 2.3055820
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2382121, 1.2331345
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6406350, 1.5951538

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881560
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881561
time: 4.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911618
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911613
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6881540, upper bound: 0.6911610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885153
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885145
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6908019, upper bound: 0.6885145
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6880497, upper bound: 0.6911631
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885138
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885133
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885138
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6906032, upper bound: 0.6885134
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906034
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906037
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906035
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6906036
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6880518
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908021
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908023
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908021
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6885133, upper bound: 0.6908023
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881560
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881561
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881560
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 5, lower bound: -0.6911611, upper bound: 0.6881561

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7579494, 1.6705356
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0129323, 1.9266801
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8120527, 1.7848392
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6380749, 1.5610161
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8731070, 1.8977776
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0979623, 1.1047565
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7219424, 1.6832039
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2596941, 2.2472343
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1745174, 1.1893991
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5596085, 1.5873957

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891383
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6862036, upper bound: 0.6833835
time: 6.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6968775, 1.7380280
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9768314, 1.9710882
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8101840, 1.7946174
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6082716, 1.6097648
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8968015, 1.8770828
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1029757, 1.1018069
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7047057, 1.7067914
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2505960, 2.2585692
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1944361, 1.1706223
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5809431, 1.5773077

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891392
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6862036, upper bound: 0.6833854
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7433348, 1.6949532
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9998493, 1.9433606
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8141189, 1.7841349
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6335630, 1.5731385
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8768511, 1.9031062
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1013321, 1.1088271
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7237134, 1.6913712
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2633915, 2.2463503
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1797614, 1.1920502
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5701747, 1.5956879

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891392
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6862036, upper bound: 0.6833834
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6833787, 1.7624457
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9721446, 1.9877687
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8122501, 1.7937984
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6027431, 1.6218874
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9005456, 1.8828712
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1043235, 1.1058774
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7074604, 1.7149589
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2542925, 2.2588334
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1996801, 1.1728368
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5915089, 1.5872564

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891392
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6862036, upper bound: 0.6833854
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7516408, 1.6826942
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9879422, 1.9705172
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7912407, 1.8137317
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6116247, 1.6090052
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8830414, 1.8912745
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0981988, 1.1045620
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7167063, 1.6976891
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2593832, 2.2509322
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1734688, 1.1911528
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5861306, 1.5743694

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864892
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888516, upper bound: 0.6807375
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6889009, 1.7426498
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9465303, 1.9982209
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7815795, 1.8158956
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5654507, 1.6398270
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9032807, 1.8671246
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1031440, 1.1015701
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6907139, 1.7139421
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2468996, 2.2595267
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1926827, 1.1717749
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5945621, 1.5542889

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864889
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888516, upper bound: 0.6807375
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7387133, 1.7071118
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9727187, 1.9871976
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7933068, 1.8127389
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6035004, 1.6211278
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8867855, 1.8966269
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1015686, 1.1086326
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7165604, 1.7058563
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2630796, 2.2500467
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1787128, 1.1938035
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5966969, 1.5820684

Time for backsubstitution: 5.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864890
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888516, upper bound: 0.6807375
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6712198, 1.7670677
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9283094, 2.0149014
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7836456, 1.8146074
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5547533, 1.6519494
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9070249, 1.8729362
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1045178, 1.1056406
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6929727, 1.7221093
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2505960, 2.2591453
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1979268, 1.1738852
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6051278, 1.5607333

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864887
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888516, upper bound: 0.6807375
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7591343, 1.6711841
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0138679, 1.9257240
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8144317, 1.7795985
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6355743, 1.5623252
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8751798, 1.9045777
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0975767, 1.1049163
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7230420, 1.6838925
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2591448, 2.2513585
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1751351, 1.1902283
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5631824, 1.5860052

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891383
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6861958, upper bound: 0.6833854
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6980624, 1.7386770
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9777675, 1.9701321
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8125625, 1.7893748
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6057701, 1.6110742
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8988705, 1.8838835
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1025904, 1.1019667
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7058053, 1.7074802
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2500467, 2.2626934
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1950538, 1.1714512
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5845165, 1.5759177

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891392
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6861958, upper bound: 0.6833835
time: 4.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7445202, 1.6956017
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0007854, 1.9424045
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8164983, 1.7788944
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6310606, 1.5744476
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8789239, 1.9099069
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1009468, 1.1089871
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7248130, 1.6920605
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2628412, 2.2504745
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1803789, 1.1928794
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5737481, 1.5942974

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891392
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6861958, upper bound: 0.6833854
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6845636, 1.7630947
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9730802, 1.9868126
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8146291, 1.7885556
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6002388, 1.6231964
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9026146, 1.8896718
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1039382, 1.1060375
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7085595, 1.7156479
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2537432, 2.2629576
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.2002976, 1.1736656
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5950828, 1.5858665

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6891392
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6861958, upper bound: 0.6833835
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7528257, 1.6833427
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9888778, 1.9695611
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7936196, 1.8084915
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6091242, 1.6103146
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8851147, 1.8980746
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0978135, 1.1047219
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7178059, 1.6983776
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2588329, 2.2550564
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1740866, 1.1919820
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5897045, 1.5729795

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864893
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888438, upper bound: 0.6807353
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6900864, 1.7432988
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9474664, 1.9972649
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7839580, 1.8106561
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5629492, 1.6411362
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9053497, 1.8739252
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1027588, 1.1017301
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6918135, 1.7146306
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2463503, 2.2636509
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1933005, 1.1726037
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5981359, 1.5528994

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864888
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888438, upper bound: 0.6807353
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7398982, 1.7077603
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9736543, 1.9862416
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7956862, 1.8074989
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6009979, 1.6224370
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8888588, 1.9034276
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1011833, 1.1087927
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7176600, 1.7065456
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2625294, 2.2541709
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1793303, 1.1946328
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6002707, 1.5806780

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864895
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888438, upper bound: 0.6807353
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6724052, 1.7677166
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9292455, 2.0139453
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7860246, 1.8093677
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5522499, 1.6532586
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.9090939, 1.8797369
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1041328, 1.1058009
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6940722, 1.7227986
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2500467, 2.2632694
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1985443, 1.1747141
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.6087017, 1.5593438

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6833833, upper bound: 0.6864887
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6888438, upper bound: 0.6807353
time: 6.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7584620, 1.6706245
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -2.0103469, 1.9310241
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8158054, 1.7875242
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6462421, 1.5671046
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8714418, 1.9013515
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1007754, 1.1041328
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7101235, 1.6948218
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2596951, 2.2472467
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1746211, 1.1892023
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5631680, 1.5876594

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6888459
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6864891, upper bound: 0.6833847
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6973906, 1.7381170
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9742470, 1.9754322
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8139367, 1.7973025
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6164389, 1.6158533
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8951368, 1.8806567
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1057888, 1.1011834
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6928868, 1.7184093
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2505960, 2.2585812
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1945398, 1.1704257
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5845027, 1.5775714

Time for backsubstitution: 5.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1922

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807354, upper bound: 0.6888459
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6864891, upper bound: 0.6833854
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.7438478, 1.6950421
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9972653, 1.9477036
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.8178720, 1.7868202
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6417294, 1.5792274
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.8751860, 1.9066801
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.1041452, 1.1082034
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.7118945, 1.7029891
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.2633915, 2.2463627
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1798654, 1.1918534
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5737348, 1.5959520

Time for backsubstitution: 5.40 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.139000654220581
rel_dist={5: [-0.78768892317715, 0.7876895182364123]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2216
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2216

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6980176, upper bound: 0.6987229
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6987208, upper bound: 0.6980196
time: 3.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 5, lower bound: -0.6980176, upper bound: 0.6987229
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 5, lower bound: -0.6987208, upper bound: 0.6980196

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6490159, 1.6489868
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9305573, 1.9284883
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7695360, 1.7673879
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6018829, 1.6088958
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7821484, 1.7803540
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0698401, 1.0703392
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6432791, 1.6345844
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1860399, 2.1860409
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1702714, 1.1704288
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5640445, 1.5638332

Time for backsubstitution: 5.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6856064, upper bound: 0.6858264
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6856064, upper bound: 0.6858264
time: 3.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6489868, 1.6490159
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9284887, 1.9305568
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7673879, 1.7695358
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6088953, 1.6018827
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7803540, 1.7821488
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0703391, 1.0698402
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6345844, 1.6432791
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1860409, 2.1860404
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1704288, 1.1702714
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5638328, 1.5640450

Time for backsubstitution: 5.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6858243, upper bound: 0.6856084
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6858243, upper bound: 0.6856084
time: 3.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.83
Output dim: 5, lower bound: -0.6856064, upper bound: 0.6858264
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.83
Output dim: 5, lower bound: -0.6856064, upper bound: 0.6858264
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.83
Output dim: 5, lower bound: -0.6858243, upper bound: 0.6856084
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.83
Output dim: 5, lower bound: -0.6858243, upper bound: 0.6856084

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6478310, 1.6472828
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9296207, 1.9283173
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7671566, 1.7692013
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6025763, 1.6075859
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7736902, 1.7735534
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0699883, 1.0701790
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6417112, 1.6338964
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1823568, 2.1819172
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1696537, 1.1691480
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5604711, 1.5613708

Time for backsubstitution: 5.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6741613, upper bound: 0.6809111
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807105, upper bound: 0.6743633
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6490159, 1.6478019
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9305573, 1.9275525
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7695360, 1.7650084
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6005726, 1.6088958
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7753482, 1.7803540
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0696800, 1.0703392
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6425910, 1.6345844
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1819172, 2.1860409
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1702714, 1.1698111
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5640445, 1.5602593

Time for backsubstitution: 5.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6741613, upper bound: 0.6809111
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6807105, upper bound: 0.6743633
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6478019, 1.6473541
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9275522, 1.9296818
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7650084, 1.7713492
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6091089, 1.6005728
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7723579, 1.7753482
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0704014, 1.0696801
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6322579, 1.6425910
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1823568, 2.1819172
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1698112, 1.1689906
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5602593, 1.5615821

Time for backsubstitution: 5.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6743625, upper bound: 0.6807126
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6809091, upper bound: 0.6741612
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6489868, 1.6478307
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9284887, 1.9296207
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7673879, 1.7671566
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.6075859, 1.6018827
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7735538, 1.7821488
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0701790, 1.0698402
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6338964, 1.6432791
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1819172, 2.1860404
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1704288, 1.1696537
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5638328, 1.5604711

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1689
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1689

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6743625, upper bound: 0.6807126
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6809091, upper bound: 0.6741612
time: 5.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6741613, upper bound: 0.6809111
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6807105, upper bound: 0.6743633
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6741613, upper bound: 0.6809111
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6807105, upper bound: 0.6743633
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6743625, upper bound: 0.6807126
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6809091, upper bound: 0.6741612
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6743625, upper bound: 0.6807126
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.53
Output dim: 5, lower bound: -0.6809091, upper bound: 0.6741612

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6341424, 1.6238678
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9127364, 1.8763645
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7436275, 1.7227883
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5643940, 1.5310123
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7647872, 1.7725987
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0696084, 1.0699549
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6284814, 1.6090763
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1813498, 2.1779532
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1671793, 1.1658349
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5353169, 1.5574350

Time for backsubstitution: 5.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647830
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647830
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6244154, 1.6335948
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8776679, 1.9114342
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7207441, 1.7456715
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5260019, 1.5694039
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7727351, 1.7646508
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0697641, 1.0697994
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6168909, 1.6206644
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1783924, 2.1809101
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1663406, 1.1666737
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5565343, 1.5362167

Time for backsubstitution: 5.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6353278, 1.6243868
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9136715, 1.8755996
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7460070, 1.7185957
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5623903, 1.5323226
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7664456, 1.7793994
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0693004, 1.0701151
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6293607, 1.6097655
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1809101, 2.1820765
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1677966, 1.1664979
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5388899, 1.5563231

Time for backsubstitution: 5.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647831
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647831
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6256008, 1.6341136
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8786039, 1.9106693
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7231226, 1.7414794
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5239992, 1.5707140
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7743936, 1.7714515
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0694559, 1.0699595
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6177707, 1.6213536
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1779528, 2.1850338
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1669579, 1.1673367
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5601077, 1.5351048

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6341133, 1.6239390
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9106693, 1.8777289
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7414799, 1.7249362
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5709267, 1.5239990
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7634554, 1.7743936
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0700216, 1.0694560
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6190262, 1.6177707
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1813498, 2.1779528
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1673367, 1.1656774
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5351048, 1.5576458

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
time: 5.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6243868, 1.6336660
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8755994, 1.9127970
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7185955, 1.7478197
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5325356, 1.5623906
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7714033, 1.7664456
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0701773, 1.0693004
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6074381, 1.6293607
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1783934, 2.1809101
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1664979, 1.1665162
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5563231, 1.5364275

Time for backsubstitution: 5.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6352987, 1.6244156
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9116049, 1.8776679
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7438583, 1.7207439
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5694036, 1.5253093
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7646508, 1.7811942
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0697992, 1.0696161
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6206646, 1.6184599
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1809101, 2.1820765
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1679542, 1.1663406
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5386782, 1.5565348

Time for backsubstitution: 5.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
time: 6.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6255717, 1.6341424
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8765354, 1.9127362
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7209749, 1.7436275
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5310125, 1.5637009
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7725987, 1.7732463
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0699549, 1.0694605
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6090765, 1.6300499
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1779528, 2.1850333
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1671152, 1.1671793
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5598965, 1.5353165

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1102
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 1102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
time: 4.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647830
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647830
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647831
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6603809, upper bound: 0.6647831
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6645067, upper bound: 0.6606544
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6606523, upper bound: 0.6645067
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.10
Output dim: 5, lower bound: -0.6647811, upper bound: 0.6603829

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6238332, 1.6027596
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8993301, 1.8592091
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7415619, 1.7212858
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5567427, 1.5189385
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7613254, 1.7645063
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0643394, 1.0657640
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6179805, 1.6007788
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1776533, 2.1749635
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1617749, 1.1586589
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5236835, 1.5378428

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6130342, 1.6238678
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8955808, 1.8763645
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7436275, 1.7207224
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5523195, 1.5310123
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7647872, 1.7691369
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0654175, 1.0699549
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6201839, 1.6090763
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1813498, 2.1742563
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1671793, 1.1604304
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5353169, 1.5458016

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6174521, 1.6124866
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8750892, 1.8942788
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7186785, 1.7443998
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5224857, 1.5573299
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7692733, 1.7565398
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0644741, 1.0656084
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6067867, 1.6123672
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1746960, 2.1779222
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1609361, 1.1595808
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5449014, 1.5194283

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207353
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207353
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6033072, 1.6335948
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8605127, 1.9114342
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7207441, 1.7436056
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5139284, 1.5694039
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7727351, 1.7611890
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0655732, 1.0697994
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6085939, 1.6206644
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1783924, 2.1772137
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1663406, 1.1612692
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5565343, 1.5245833

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207338
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207338
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6250181, 1.6032786
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.9002657, 1.8584442
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7439404, 1.7170932
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5547419, 1.5202472
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7629838, 1.7713070
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0640311, 1.0659238
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6188602, 1.6014664
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1772137, 2.1790872
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1623921, 1.1593219
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5272570, 1.5367308

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247396
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247396
time: 4.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6142192, 1.6243868
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8965168, 1.8755996
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7460070, 1.7165298
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5503168, 1.5323226
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7664456, 1.7759376
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0651093, 1.0701151
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6210632, 1.6097655
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1809101, 2.1783800
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1677966, 1.1610935
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5388899, 1.5446901

Time for backsubstitution: 5.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247395
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247395
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6186376, 1.6130054
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8760252, 1.8935139
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7210569, 1.7402077
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5204849, 1.5586388
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7709312, 1.7633405
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0641658, 1.0657682
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6076665, 1.6130548
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1742563, 2.1820455
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1615534, 1.1602440
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5484748, 1.5183163

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6044927, 1.6341136
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8614483, 1.9106693
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7231226, 1.7394135
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5119257, 1.5707140
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7743936, 1.7679896
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0652649, 1.0699595
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6094732, 1.6213536
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1779528, 2.1813374
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1669579, 1.1619322
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5601077, 1.5234718

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6238041, 1.6028309
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8972626, 1.8605735
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7394133, 1.7234340
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5632763, 1.5119252
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7599936, 1.7663012
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0647523, 1.0652649
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6085253, 1.6094735
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1776533, 2.1749630
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1619322, 1.1585014
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5234718, 1.5380540

Time for backsubstitution: 5.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241993
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241984
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6130052, 1.6239390
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8935142, 1.8777289
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7414799, 1.7228706
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5588531, 1.5239990
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7634554, 1.7709312
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0658307, 1.0694560
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6107292, 1.6177707
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1813498, 2.1742563
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1673367, 1.1602730
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5351048, 1.5460129

Time for backsubstitution: 5.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241988
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241990
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6174231, 1.6125579
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8730202, 1.8956418
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7165298, 1.7465479
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5290194, 1.5503168
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7679415, 1.7583342
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0648870, 1.0651094
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5973330, 1.6210632
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1746960, 2.1779218
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1610935, 1.1594234
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5446897, 1.5196390

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6032786, 1.6336660
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8584442, 1.9127970
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7185955, 1.7457538
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5204620, 1.5623906
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7714033, 1.7629838
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0659862, 1.0693004
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5991406, 1.6293607
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1783934, 2.1772132
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1664979, 1.1611117
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5563231, 1.5247946

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6249890, 1.6033075
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8981981, 1.8605127
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7417917, 1.7192414
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5617542, 1.5132339
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7611890, 1.7731013
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0645299, 1.0654249
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6101632, 1.6101611
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1772137, 2.1790867
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1625497, 1.1591644
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5270452, 1.5369425

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244144
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244135
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6141906, 1.6244156
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8944492, 1.8776679
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7438583, 1.7186780
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5573301, 1.5253093
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7646508, 1.7777319
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0656083, 1.0696161
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6123667, 1.6184599
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1809101, 2.1783795
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1679542, 1.1609361
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5386782, 1.5449014

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244139
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244141
time: 5.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6186080, 1.6130342
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8739557, 1.8955808
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7189083, 1.7423558
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5274973, 1.5516255
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7691369, 1.7651348
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0646648, 1.0652692
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.5989714, 1.6217508
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1742563, 2.1820450
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1617107, 1.1600865
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5482635, 1.5185285

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1630764, -10.4619446, -13.1630764, -10.4619446, -1.6044636, 1.6341424
1: -11.3005276, -8.3500328, -11.3005276, -8.3500328, -1.8593798, 1.9127362
2: -10.7255383, -8.5164833, -10.7255383, -8.5164833, -1.7209749, 1.7415617
3: -4.4308734, -2.2690167, -4.4308734, -2.2690167, -1.5189381, 1.5637009
4: -15.1825619, -12.4986038, -15.1825619, -12.4986038, -1.7725987, 1.7697840
5: 8.2191973, 9.7242146, 8.2191973, 9.7242146, -1.0657640, 1.0694605
6: -4.7438617, -2.2754083, -4.7438617, -2.2754083, -1.6007786, 1.6300499
7: -15.7447653, -12.9006634, -15.7447653, -12.9006634, -2.1779528, 2.1813369
8: -0.8548889, 0.9188228, -0.8548889, 0.9188228, -1.1671152, 1.1617749
9: -6.7094598, -5.0025444, -6.7094598, -5.0025444, -1.5598965, 1.5236835

Time for backsubstitution: 5.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2375
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 425
type: RSZ, layer: 3, pos: 2629
type: RSZ, layer: 3, pos: 2537
type: RSZ, layer: 3, pos: 1948
type: RSZ, layer: 3, pos: 2805
type: RSZ, layer: 3, pos: 1446
type: RSZ, layer: 3, pos: 2853
type: RSZ, layer: 3, pos: 410
type: RSZ, layer: 3, pos: 2627
type: RSZ, layer: 3, pos: 1836
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2530
type: RSZ, layer: 3, pos: 2914
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1829
type: RSZ, layer: 3, pos: 1412
type: RSZ, layer: 3, pos: 2468
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3127
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 1397
type: RSZ, layer: 3, pos: 1194
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 429
type: RSZ, layer: 3, pos: 2831
type: RSZ, layer: 3, pos: 550
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 158
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 961
type: RSZ, layer: 3, pos: 424
type: RSZ, layer: 3, pos: 918
type: RSZ, layer: 3, pos: 232
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1773
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 949
type: RSZ, layer: 3, pos: 1835
type: RSZ, layer: 3, pos: 976

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 2375

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
time: 4.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6204098, upper bound: 0.6247378
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207353
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207338
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6244141, upper bound: 0.6207338
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247396
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247396
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247395
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6202567, upper bound: 0.6247395
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6241989, upper bound: 0.6207353
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241993
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241984
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241988
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6241990
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6202587
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244144
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244135
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244139
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6207333, upper bound: 0.6244141
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.56
Output dim: 5, lower bound: -0.6247377, upper bound: 0.6204118
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.0726536512374878
rel_dist={5: [-0.7059536409330516, 0.7059557061852892]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.015625
execution time: 2122.57 seconds
