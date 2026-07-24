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
execution time: IAR + LP analysis = 14.91 + 32.18 = 47.09 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.91 seconds, max iter: 100)

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
Binary search time: 152.05 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 3400.86 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8402140, upper bound: 1.8405107
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8405112, upper bound: 1.8402133
time: 5.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.53
Output dim: 6, lower bound: -1.8402140, upper bound: 1.8405107
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.53
Output dim: 6, lower bound: -1.8405112, upper bound: 1.8402133

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6551018, 3.6549578
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -4.0231352, 4.0158324
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.5168037, 4.5071387
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4319043, 3.4331326
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5203047, 3.5175352
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9914408, 3.9969163
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.0147486, 3.0103774

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8401971, upper bound: 1.8351062
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8348137, upper bound: 1.8404937
time: 4.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6549578, 3.6551018
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -4.0158319, 4.0231352
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.5071392, 4.5168028
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4331326, 3.4319043
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5175352, 3.5203052
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9969168, 3.9914408
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.0103769, 3.0147495

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404941, upper bound: 1.8348134
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351065, upper bound: 1.8401969
time: 4.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 6, lower bound: -1.8401971, upper bound: 1.8351062
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 6, lower bound: -1.8348137, upper bound: 1.8404937
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 6, lower bound: -1.8404941, upper bound: 1.8348134
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.86
Output dim: 6, lower bound: -1.8351065, upper bound: 1.8401969

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6540194, 3.6546097
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9983568, 4.0076609
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.5134201, 4.4967318
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4312582, 3.4311552
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5199471, 3.5163455
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9862442, 3.9810834
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.0083780, 2.9910610

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8401963, upper bound: 1.8340837
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8391937, upper bound: 1.8351054
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6547537, 3.6538749
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -4.0149641, 3.9910531
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.5063953, 4.5037551
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4299269, 3.4324870
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5191154, 3.5171781
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9756088, 3.9917197
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9954329, 3.0040052

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8348129, upper bound: 1.8394701
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8338119, upper bound: 1.8404931
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6538744, 3.6547537
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9910536, 4.0149641
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.5037556, 4.5063958
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4324865, 3.4299269
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5171776, 3.5191159
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9917202, 3.9756074
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.0040045, 2.9954331

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404933, upper bound: 1.8338112
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8394704, upper bound: 1.8348126
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6546087, 3.6540194
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -4.0076609, 3.9983559
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4967327, 4.5134192
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4311552, 3.4312587
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5163460, 3.5199475
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9810829, 3.9862442
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9910612, 3.0083773

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8351057, upper bound: 1.8391935
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8340839, upper bound: 1.8401961
time: 4.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.37 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8401963, upper bound: 1.8340837
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8391937, upper bound: 1.8351054
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8348129, upper bound: 1.8394701
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8338119, upper bound: 1.8404931
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8404933, upper bound: 1.8338112
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8394704, upper bound: 1.8348126
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8351057, upper bound: 1.8391935
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.37
Output dim: 6, lower bound: -1.8340839, upper bound: 1.8401961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6353216, 3.6274734
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9168653, 3.8986354
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4998150, 4.4882107
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4271755, 3.4280934
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5385504, 3.5260720
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9711370, 3.9724798
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9948187, 2.9681537

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8401886, upper bound: 1.8268700
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330570, upper bound: 1.8340755
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6268826, 3.6359124
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8893309, 3.9261694
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.5048981, 4.4831276
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4281969, 3.4270725
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5296736, 3.5349488
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9776411, 3.9659753
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9854708, 2.9775026

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8391859, upper bound: 1.8278950
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8320387, upper bound: 1.8350974
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6360569, 3.6267390
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9334726, 3.8820276
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4927921, 4.4952340
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4258442, 3.4294252
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5377188, 3.5269046
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9604998, 3.9831161
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9818754, 2.9810979

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8348051, upper bound: 1.8322650
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8276612, upper bound: 1.8394621
time: 6.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6276178, 3.6351776
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9059381, 3.9095616
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4978752, 4.4901509
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4268656, 3.4284039
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5288420, 3.5357804
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9670048, 3.9766121
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9725256, 2.9904468

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8338041, upper bound: 1.8332901
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8266515, upper bound: 1.8404857
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6351776, 3.6276174
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9095621, 3.9059381
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4901505, 4.4978747
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4284039, 3.4268651
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5357809, 3.5288424
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9766121, 3.9670043
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9904470, 2.9725258

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404857, upper bound: 1.8266512
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332902, upper bound: 1.8338038
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6267385, 3.6360564
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8820276, 3.9334722
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4952335, 4.4927917
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4294252, 3.4258442
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5269041, 3.5377188
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9831171, 3.9604998
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9810972, 2.9818747

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8394626, upper bound: 1.8276609
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8322650, upper bound: 1.8348049
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6359119, 3.6268830
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9261694, 3.8893304
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4831276, 4.5048981
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4270725, 3.4281969
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5349474, 3.5296741
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9659758, 3.9776411
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9775019, 2.9854701

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350979, upper bound: 1.8320385
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278953, upper bound: 1.8391854
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6274738, 3.6353221
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8986359, 3.9168644
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4882107, 4.4998150
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4280930, 3.4271755
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5260725, 3.5385509
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9724798, 3.9711361
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9681540, 2.9948189

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8340761, upper bound: 1.8330567
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8268701, upper bound: 1.8401883
time: 4.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8401886, upper bound: 1.8268700
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8330570, upper bound: 1.8340755
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8391859, upper bound: 1.8278950
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8320387, upper bound: 1.8350974
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8348051, upper bound: 1.8322650
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8276612, upper bound: 1.8394621
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8338041, upper bound: 1.8332901
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8266515, upper bound: 1.8404857
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8404857, upper bound: 1.8266512
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8332902, upper bound: 1.8338038
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8394626, upper bound: 1.8276609
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8322650, upper bound: 1.8348049
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8350979, upper bound: 1.8320385
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8278953, upper bound: 1.8391854
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8340761, upper bound: 1.8330567
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.86
Output dim: 6, lower bound: -1.8268701, upper bound: 1.8401883

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6301780, 3.6281524
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8745742, 3.8751822
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4646721, 4.4413524
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4153891, 3.4109073
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5322037, 3.5220571
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9418640, 3.9505200
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9199786, 2.8683853

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8382327, upper bound: 1.8268681
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8401866, upper bound: 1.8248747
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6360011, 3.6223297
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8934112, 3.8563452
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4529572, 4.4530673
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4099894, 3.4163065
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5345345, 3.5197263
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9491768, 3.9432068
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8950505, 2.8933127

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8310777, upper bound: 1.8340737
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330550, upper bound: 1.8320760
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6217389, 3.6365914
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8470397, 3.9027162
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4697552, 4.4362693
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4164095, 3.4098859
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5233269, 3.5309339
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9483681, 3.9440160
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9106288, 2.8777342

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8372314, upper bound: 1.8278931
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8391842, upper bound: 1.8258954
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6275620, 3.6307683
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8658767, 3.8838792
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4580402, 4.4479842
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4110098, 3.4152856
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5256596, 3.5286021
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9556808, 3.9367023
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8857017, 2.9026616

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8300582, upper bound: 1.8350958
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8320370, upper bound: 1.8330969
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6309123, 3.6274180
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8911815, 3.8585739
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4576473, 4.4483757
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4140577, 3.4122386
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5313721, 3.5228891
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9312267, 3.9611564
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9070334, 2.8813295

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8328461, upper bound: 1.8322630
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8348031, upper bound: 1.8302693
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6367354, 3.6215949
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9100184, 3.8397369
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4459324, 4.4600911
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4086580, 3.4176383
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5337029, 3.5205579
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9385414, 3.9538436
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8821063, 2.9062569

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8256844, upper bound: 1.8394604
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8276591, upper bound: 1.8374625
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6224732, 3.6358566
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8636470, 3.8861084
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4627304, 4.4432926
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4150782, 3.4112177
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5224953, 3.5317655
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9377327, 3.9546518
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8976855, 2.8906784

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8318436, upper bound: 1.8332880
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8338024, upper bound: 1.8312902
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6282964, 3.6300335
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8824840, 3.8672714
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4510155, 4.4550085
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4096785, 3.4166169
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5248260, 3.5294347
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9450455, 3.9473386
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8727574, 2.9156058

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8246700, upper bound: 1.8404830
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8266498, upper bound: 1.8384849
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6300330, 3.6282969
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8672709, 3.8824844
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4550076, 4.4510164
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4166174, 3.4096789
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5294342, 3.5248275
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9473400, 3.9450450
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9156060, 2.8727574

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8384854, upper bound: 1.8266494
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404836, upper bound: 1.8246697
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6358562, 3.6224737
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8861079, 3.8636475
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4432926, 4.4627309
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4112177, 3.4150782
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5317650, 3.5224962
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9546528, 3.9377313
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8906779, 2.8976851

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8312907, upper bound: 1.8338020
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332882, upper bound: 1.8318433
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6215949, 3.6367354
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8397365, 3.9100189
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4600906, 4.4459333
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4176378, 3.4086576
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5205574, 3.5337038
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9538441, 3.9385400
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9062572, 2.8821063

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8374624, upper bound: 1.8276588
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8394609, upper bound: 1.8256842
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6274180, 3.6309123
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8585734, 3.8911819
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4483757, 4.4576483
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4122381, 3.4140573
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5228882, 3.5313725
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9611568, 3.9312267
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8813300, 2.9070339

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8302698, upper bound: 1.8348027
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8322633, upper bound: 1.8328460
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6307683, 3.6275620
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8838782, 3.8658767
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4479847, 4.4580398
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4152861, 3.4110103
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5286026, 3.5256591
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9367027, 3.9556808
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9026618, 2.8857017

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330972, upper bound: 1.8320367
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350958, upper bound: 1.8300581
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6365914, 3.6217389
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9027152, 3.8470397
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4362698, 4.4697556
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4098864, 3.4164100
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5309334, 3.5233283
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9440155, 3.9483676
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8777347, 2.9106293

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8258957, upper bound: 1.8391839
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8278933, upper bound: 1.8372310
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6223292, 3.6360011
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8563457, 3.8934107
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4530678, 4.4529567
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4163065, 3.4099898
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5197258, 3.5345359
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9432068, 3.9491763
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8933129, 2.8950505

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8320764, upper bound: 1.8330551
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8340744, upper bound: 1.8310772
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6281524, 3.6301780
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8751826, 3.8745737
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4413528, 4.4646726
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4109068, 3.4153891
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5220566, 3.5322046
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9505215, 3.9418635
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8683848, 2.9199781

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248748, upper bound: 1.8401864
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8268684, upper bound: 1.8382325
time: 4.98 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8382327, upper bound: 1.8268681
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8401866, upper bound: 1.8248747
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8310777, upper bound: 1.8340737
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8330550, upper bound: 1.8320760
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8372314, upper bound: 1.8278931
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8391842, upper bound: 1.8258954
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8300582, upper bound: 1.8350958
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8320370, upper bound: 1.8330969
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8328461, upper bound: 1.8322630
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8348031, upper bound: 1.8302693
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8256844, upper bound: 1.8394604
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8276591, upper bound: 1.8374625
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8318436, upper bound: 1.8332880
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8338024, upper bound: 1.8312902
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8246700, upper bound: 1.8404830
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8266498, upper bound: 1.8384849
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8384854, upper bound: 1.8266494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8404836, upper bound: 1.8246697
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8312907, upper bound: 1.8338020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8332882, upper bound: 1.8318433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8374624, upper bound: 1.8276588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8394609, upper bound: 1.8256842
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8302698, upper bound: 1.8348027
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8322633, upper bound: 1.8328460
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8330972, upper bound: 1.8320367
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8350958, upper bound: 1.8300581
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8258957, upper bound: 1.8391839
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8278933, upper bound: 1.8372310
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8320764, upper bound: 1.8330551
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8340744, upper bound: 1.8310772
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8248748, upper bound: 1.8401864
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 6, lower bound: -1.8268684, upper bound: 1.8382325
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.050609588623047
rel_dist={6: [-1.840513682283781, 1.840511023435588]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4760095, upper bound: 1.4762388
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762367, upper bound: 1.4760115
time: 6.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.43
Output dim: 6, lower bound: -1.4760095, upper bound: 1.4762388
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.43
Output dim: 6, lower bound: -1.4762367, upper bound: 1.4760115

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4122410, 3.4121685
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7174501, 3.7137995
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1858253, 4.1809931
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8459044, 3.8438625
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2644544, 3.2650685
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2047272, 3.2033424
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8201857, 3.8229241
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7946253, 2.7924390

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4759996, upper bound: 1.4733620
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731358, upper bound: 1.4762264
time: 6.08 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4121685, 3.4122405
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7137995, 3.7174506
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1809931, 4.1858249
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8438625, 3.8459048
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2650685, 3.2644544
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2033424, 3.2047276
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8229246, 3.8201861
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7924395, 2.7946250

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762268, upper bound: 1.4731350
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733631, upper bound: 1.4759992
time: 11.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 30.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.98
Output dim: 6, lower bound: -1.4759996, upper bound: 1.4733620
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.98
Output dim: 6, lower bound: -1.4731358, upper bound: 1.4762264
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.98
Output dim: 6, lower bound: -1.4762268, upper bound: 1.4731350
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.98
Output dim: 6, lower bound: -1.4733631, upper bound: 1.4759992

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4111576, 3.4114532
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6926718, 3.6973238
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1789303, 4.1705861
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8439617, 3.8409414
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2631426, 3.2630906
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2039547, 3.2021537
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8096714, 3.8070908
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7817812, 2.7731225

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4759986, upper bound: 1.4721014
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4747388, upper bound: 1.4733609
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4115248, 3.4110861
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7009745, 3.6890202
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1754189, 4.1740975
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8429842, 3.8419189
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2624769, 3.2637568
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2035389, 3.2025695
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8043537, 3.8124089
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7753077, 2.7795947

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731348, upper bound: 1.4749657
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718751, upper bound: 1.4762246
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4110861, 3.4115252
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6890192, 3.7009754
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1740971, 4.1754179
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8419199, 3.8429842
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2637568, 3.2624764
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2025700, 3.2035384
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8124104, 3.8043532
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7795954, 2.7753086

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762258, upper bound: 1.4718739
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4749660, upper bound: 1.4731337
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4114532, 3.4111581
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6973238, 3.6926713
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1705856, 4.1789298
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8409424, 3.8439608
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2630911, 3.2631426
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2021542, 3.2039542
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8070908, 3.8096709
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7731218, 2.7817807

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733620, upper bound: 1.4718747
time: 10.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721023, upper bound: 1.4759974
time: 4.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 29.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4759986, upper bound: 1.4721014
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4747388, upper bound: 1.4733609
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4731348, upper bound: 1.4749657
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4718751, upper bound: 1.4762246
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4762258, upper bound: 1.4718739
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4749660, upper bound: 1.4731337
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4733620, upper bound: 1.4718747
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 6, lower bound: -1.4721023, upper bound: 1.4759974

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3882408, 3.3843169
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5974131, 3.5882981
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7352877, 3.7318659
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1653252, 4.1595235
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8396568, 3.8371716
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2590599, 3.2595186
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2181196, 3.2118802
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7945633, 3.7952352
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7635489, 2.7502153

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4759947, upper bound: 1.4687749
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4726766, upper bound: 1.4720982
time: 5.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3840218, 3.3885365
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5836458, 3.6020651
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7226477, 3.7445068
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1678677, 4.1569819
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8401909, 3.8366375
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2595701, 3.2590079
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2136812, 3.2163181
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7978163, 3.7919827
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7588739, 2.7548897

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4747349, upper bound: 1.4700366
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4714148, upper bound: 1.4733578
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3886089, 3.3839498
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6057177, 3.5799942
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7412748, 3.7258787
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1618137, 4.1630349
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8386803, 3.8381491
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2583933, 3.2601843
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2177019, 3.2122960
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7892456, 3.8005533
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7570753, 2.7566874

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731310, upper bound: 1.4716403
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4698112, upper bound: 1.4749617
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3843889, 3.3881693
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5919504, 3.5937612
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7286348, 3.7385197
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1643562, 4.1604934
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8392143, 3.8376150
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2589045, 3.2596741
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2132654, 3.2167344
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7924976, 3.7973013
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7524023, 2.7613618

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718712, upper bound: 1.4729021
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4685496, upper bound: 1.4762215
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3881693, 3.3843894
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5937614, 3.5919495
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7385206, 3.7286339
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1604939, 4.1643553
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8376150, 3.8392143
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2596741, 3.2589045
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2167330, 3.2132649
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7973013, 3.7924976
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7613611, 2.7524014

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762219, upper bound: 1.4685488
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729027, upper bound: 1.4718710
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3839493, 3.3886085
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5799952, 3.6057167
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7258787, 3.7412748
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1630344, 4.1618137
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8381491, 3.8386803
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2601843, 3.2583938
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2122965, 3.2177033
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8005533, 3.7892451
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7566881, 2.7570758

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4749622, upper bound: 1.4698106
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4716410, upper bound: 1.4731306
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3885365, 3.3840218
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6020651, 3.5836456
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7445078, 3.7226467
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1569824, 4.1678672
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8366375, 3.8401909
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2590075, 3.2595701
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2163172, 3.2136807
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7919836, 3.7978153
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7548895, 2.7588735

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733582, upper bound: 1.4714143
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4700374, upper bound: 1.4747349
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3843174, 3.3882413
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5882988, 3.5974126
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7318659, 3.7352877
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1595230, 4.1653256
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8371716, 3.8396568
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2595186, 3.2590599
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2118788, 3.2181191
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7952356, 3.7945633
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7502146, 2.7635479

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4720984, upper bound: 1.4726762
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4687757, upper bound: 1.4759946
time: 5.08 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4759947, upper bound: 1.4687749
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4726766, upper bound: 1.4720982
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4747349, upper bound: 1.4700366
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4714148, upper bound: 1.4733578
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4731310, upper bound: 1.4716403
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4698112, upper bound: 1.4749617
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4718712, upper bound: 1.4729021
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4685496, upper bound: 1.4762215
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4762219, upper bound: 1.4685488
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4729027, upper bound: 1.4718710
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4749622, upper bound: 1.4698106
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4716410, upper bound: 1.4731306
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4733582, upper bound: 1.4714143
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4700374, upper bound: 1.4747349
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4720984, upper bound: 1.4726762
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.07
Output dim: 6, lower bound: -1.4687757, upper bound: 1.4759946

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3830972, 3.3820848
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5551219, 3.5554261
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7216883, 3.7201362
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1243248, 4.1126647
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7855005, 3.7752829
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2445726, 3.2423320
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2117729, 3.2066994
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7652912, 3.7696190
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6762433, 2.6504469

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4748231, upper bound: 1.4687739
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4759937, upper bound: 1.4676208
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3860087, 3.3791733
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5645404, 3.5460076
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7235575, 3.7182670
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1184673, 4.1185222
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7777681, 3.7830153
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2418737, 3.2450318
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2129383, 3.2055335
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7689476, 3.7659621
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6637797, 2.6629105

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4715227, upper bound: 1.4720996
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4726756, upper bound: 1.4709256
time: 5.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3788781, 3.3863039
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5413547, 3.5691931
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7090483, 3.7327766
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1268654, 4.1101232
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7860346, 3.7747488
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2450838, 3.2418218
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2073345, 3.2111373
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7685432, 3.7663665
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6715693, 2.6551213

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4735637, upper bound: 1.4700354
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4747340, upper bound: 1.4688821
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3817897, 3.3833923
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5507731, 3.5597746
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7109175, 3.7309074
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1210079, 4.1159806
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7783022, 3.7824817
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2423830, 3.2445216
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2084999, 3.2099719
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7721996, 3.7627101
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6591048, 2.6675849

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4702504, upper bound: 1.4733569
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4714139, upper bound: 1.4721860
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3834643, 3.3817172
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5634265, 3.5471222
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7276754, 3.7141490
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1208134, 4.1161766
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7845240, 3.7762604
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2439070, 3.2429981
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2113571, 3.2071157
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7599735, 3.7749372
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6697707, 2.6569190

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4719595, upper bound: 1.4716391
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731300, upper bound: 1.4704777
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3863759, 3.3788056
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5728450, 3.5377038
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7295446, 3.7122798
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1149559, 4.1220345
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7767916, 3.7839928
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2412081, 3.2456975
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2125225, 3.2059498
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7636299, 3.7712808
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6573071, 2.6693828

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4686567, upper bound: 1.4749633
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4698103, upper bound: 1.4737904
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3792453, 3.3859367
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5496593, 3.5608892
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7150354, 3.7267900
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1233540, 4.1136351
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7850580, 3.7757263
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2444181, 3.2424874
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2069187, 3.2115536
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7632256, 3.7716846
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6650968, 2.6615934

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4706995, upper bound: 1.4729011
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718703, upper bound: 1.4717474
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3821568, 3.3830252
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5590777, 3.5514708
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7169046, 3.7249203
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1174965, 4.1194930
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7773256, 3.7834587
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2417173, 3.2451873
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2080841, 3.2103877
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7668819, 3.7680283
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6526332, 2.6740572

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4673947, upper bound: 1.4762208
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4685487, upper bound: 1.4750496
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3830256, 3.3821568
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5514712, 3.5590775
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7249212, 3.7169042
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1194935, 4.1174970
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7834587, 3.7773256
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2451868, 3.2417178
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2103882, 3.2080846
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7680283, 3.7668810
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6740575, 2.6526330

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750503, upper bound: 1.4685479
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4762209, upper bound: 1.4673940
time: 7.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3859363, 3.3792453
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5608897, 3.5496590
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7267904, 3.7150345
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1136341, 4.1233544
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7757263, 3.7850580
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2424879, 3.2444177
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2115536, 3.2069187
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7716846, 3.7632241
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6615939, 2.6650968

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4717482, upper bound: 1.4718702
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729017, upper bound: 1.4706986
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3788056, 3.3863764
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5377040, 3.5728445
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7122803, 3.7295446
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1220341, 4.1149554
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7839928, 3.7767916
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2456980, 3.2412076
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2059498, 3.2125225
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7712803, 3.7636285
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6693826, 2.6573074

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4737910, upper bound: 1.4698094
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4749613, upper bound: 1.4686555
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3817172, 3.3834648
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5471225, 3.5634260
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7141495, 3.7276750
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1161766, 4.1208129
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7762604, 3.7845240
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2429972, 3.2439075
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2071152, 3.2113571
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7749367, 3.7599721
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6569190, 2.6697712

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4704781, upper bound: 1.4731298
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4716401, upper bound: 1.4719586
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3833919, 3.3817892
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5597739, 3.5507736
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7309084, 3.7109170
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1159801, 4.1210084
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7824812, 3.7783022
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2445211, 3.2423840
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2099724, 3.2085004
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7627106, 3.7721992
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6675849, 2.6591051

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721867, upper bound: 1.4714130
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733572, upper bound: 1.4702498
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3863044, 3.3788776
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5691924, 3.5413551
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7327776, 3.7090473
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1101227, 4.1268663
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7747488, 3.7860346
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2418222, 3.2450833
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2111378, 3.2073350
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7663670, 3.7685428
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6551213, 2.6715689

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4688829, upper bound: 1.4747336
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4700364, upper bound: 1.4735630
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3791728, 3.3860087
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5460067, 3.5645406
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7182674, 3.7235575
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1185226, 4.1184669
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7830153, 3.7777681
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2450323, 3.2418733
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2055340, 3.2129388
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7659626, 3.7689471
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6629100, 2.6637795

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4709261, upper bound: 1.4726750
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4720975, upper bound: 1.4715222
time: 4.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4748231, upper bound: 1.4687739
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4759937, upper bound: 1.4676208
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4715227, upper bound: 1.4720996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4726756, upper bound: 1.4709256
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4735637, upper bound: 1.4700354
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4747340, upper bound: 1.4688821
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4702504, upper bound: 1.4733569
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4714139, upper bound: 1.4721860
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4719595, upper bound: 1.4716391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4731300, upper bound: 1.4704777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4686567, upper bound: 1.4749633
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4698103, upper bound: 1.4737904
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4706995, upper bound: 1.4729011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4718703, upper bound: 1.4717474
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4673947, upper bound: 1.4762208
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4685487, upper bound: 1.4750496
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4750503, upper bound: 1.4685479
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4762209, upper bound: 1.4673940
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4717482, upper bound: 1.4718702
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4729017, upper bound: 1.4706986
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4737910, upper bound: 1.4698094
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4749613, upper bound: 1.4686555
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4704781, upper bound: 1.4731298
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4716401, upper bound: 1.4719586
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4721867, upper bound: 1.4714130
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4733572, upper bound: 1.4702498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4688829, upper bound: 1.4747336
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4700364, upper bound: 1.4735630
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4709261, upper bound: 1.4726750
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 6, lower bound: -1.4720975, upper bound: 1.4715222
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.90
Output dim: 6, lower bound: -1.4687757, upper bound: 1.4759946
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.050609588623047
rel_dist={6: [-1.4762376626074394, 1.476239686090869]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2040510, upper bound: 1.2041191
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2040510, upper bound: 1.2040480
time: 5.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 6, lower bound: -1.2040510, upper bound: 1.2041191
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.89
Output dim: 6, lower bound: -1.2040510, upper bound: 1.2040480

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2503328, 3.2503090
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5136604, 3.5124435
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7084303, 3.7095070
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9651737, 3.9635620
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6577988, 3.6571183
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1528211, 3.1530256
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9671488, 2.9665890
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9943428, 2.9938812
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7060156, 3.7069287
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6478758, 2.6471467

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2040440, upper bound: 1.2032561
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2031829, upper bound: 1.2041144
time: 9.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2503090, 3.2503328
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5124435, 3.5136607
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7095079, 3.7084298
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9635620, 3.9651728
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6571188, 3.6577988
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1530252, 3.1528211
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9665890, 2.9671488
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9938812, 2.9943428
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7069292, 3.7060161
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6471472, 2.6478753

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2031848
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2032542, upper bound: 1.2040434
time: 10.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 31.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.60
Output dim: 6, lower bound: -1.2040440, upper bound: 1.2032561
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.60
Output dim: 6, lower bound: -1.2031829, upper bound: 1.2041144
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.60
Output dim: 6, lower bound: -1.2041152, upper bound: 1.2031848
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.60
Output dim: 6, lower bound: -1.2032542, upper bound: 1.2040434

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2492504, 3.2493486
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4888811, 3.4904323
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6905622, 3.6936355
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9559364, 3.9531550
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6552038, 3.6541977
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1510649, 3.1510482
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9664125, 2.9657612
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9932919, 2.9926920
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6919556, 3.6910958
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6307154, 2.6278303

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2040438, upper bound: 1.2028607
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2036504, upper bound: 1.2032559
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2493725, 3.2492266
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4916487, 3.4876642
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6925592, 3.6916394
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9547672, 3.9543262
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6548786, 3.6545229
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1508436, 3.1512699
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9663200, 2.9658527
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9931545, 2.9928308
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6901836, 3.6928687
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6285582, 2.6299877

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2031826, upper bound: 1.2037215
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2027875, upper bound: 1.2041138
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2492266, 3.2493730
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4876642, 3.4916494
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6916399, 3.6925578
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9543266, 3.9547658
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6545238, 3.6548781
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1512690, 3.1508431
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9658527, 2.9663205
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9928303, 2.9931536
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6928692, 3.6901832
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6299868, 2.6285589

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041149, upper bound: 1.2027875
time: 11.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037215, upper bound: 1.2031846
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2493486, 3.2492504
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4904318, 3.4888813
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6936369, 3.6905622
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9531555, 3.9559364
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6541977, 3.6552038
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1510477, 3.1510653
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9657612, 2.9664125
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9926929, 2.9932923
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6910954, 3.6919560
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6278296, 2.6307163

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2032539, upper bound: 1.2036525
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2028588, upper bound: 1.2040435
time: 7.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2040438, upper bound: 1.2028607
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2036504, upper bound: 1.2032559
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2031826, upper bound: 1.2037215
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2027875, upper bound: 1.2041138
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2041149, upper bound: 1.2027875
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2037215, upper bound: 1.2031846
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2032539, upper bound: 1.2036525
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.19
Output dim: 6, lower bound: -1.2028588, upper bound: 1.2040435

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2235212, 3.2222128
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3844452, 3.3814065
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5937033, 3.5925617
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9423332, 3.9403982
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6508999, 3.6500716
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1469822, 3.1471353
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9756684, 2.9754858
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0044985, 3.0024185
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6768484, 3.6770720
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6093664, 2.6049230

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2040426, upper bound: 1.2016872
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2028805, upper bound: 1.2028567
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2222366, 3.2234969
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3826246, 3.3832276
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5914850, 3.5947800
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9420090, 3.9407220
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6507521, 3.6502190
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1469307, 3.1471872
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9760451, 2.9751086
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0028791, 3.0040364
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6761599, 3.6777606
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6056509, 2.6086385

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2027864, upper bound: 1.2029505
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2016171, upper bound: 1.2041126
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2234974, 3.2222371
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3832283, 3.3826237
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5947809, 3.5914845
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9407215, 3.9420090
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6502190, 3.6507521
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1471872, 3.1469307
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9751086, 2.9760451
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0040369, 3.0028801
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6777611, 3.6761594
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6086378, 2.6056516

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041138, upper bound: 1.2016161
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2029519, upper bound: 1.2027850
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2222128, 3.2235208
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3814068, 3.3844447
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5925627, 3.5937023
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9403992, 3.9423323
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6500711, 3.6508999
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1471357, 3.1469822
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9754853, 2.9756680
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0024176, 3.0044980
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6770725, 3.6768484
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6049223, 2.6093671

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2028576, upper bound: 1.2028795
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2016884, upper bound: 1.2040413
time: 4.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2040426, upper bound: 1.2016872
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2028805, upper bound: 1.2028567
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2027864, upper bound: 1.2029505
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2016171, upper bound: 1.2041126
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2041138, upper bound: 1.2016161
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2029519, upper bound: 1.2027850
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2028576, upper bound: 1.2028795
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 6, lower bound: -1.2016884, upper bound: 1.2040413

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2183766, 3.2180395
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3421550, 3.3422556
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5801039, 3.5795856
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8974266, 3.8935399
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5915890, 3.5881829
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1306963, 3.1299491
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9940319, 2.9952884
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9981518, 2.9964604
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6475754, 3.6490178
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5137534, 2.5051546

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2037162, upper bound: 1.2016871
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2012724, upper bound: 1.2013399
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2180629, 3.2183528
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3434730, 3.3409371
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5785093, 3.5811801
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8951511, 3.8958163
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5888634, 3.5909081
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1297445, 3.1309009
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9958477, 2.9934721
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9969215, 2.9976902
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6481056, 3.6484876
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5058837, 2.5130248

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2012704, upper bound: 1.2041133
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2016169, upper bound: 1.2037857
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2183528, 3.2180634
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3409362, 3.3434727
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5811806, 3.5785084
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8958168, 3.8951507
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5909081, 3.5888634
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1309004, 3.1297441
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9934721, 2.9958482
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9976902, 2.9969225
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6484890, 3.6481051
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5130248, 2.5058832

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037870, upper bound: 1.2016156
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041135, upper bound: 1.2012694
time: 8.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2180390, 3.2183771
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3422561, 3.3421543
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5795860, 3.5801029
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8935394, 3.8974266
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5881834, 3.5915890
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1299486, 3.1306958
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9952888, 2.9940319
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9964600, 2.9981523
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6490192, 3.6475749
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5051551, 2.5137534

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2013406, upper bound: 1.2040419
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2016882, upper bound: 1.2037145
time: 5.26 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2037162, upper bound: 1.2016871
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2012724, upper bound: 1.2013399
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2012704, upper bound: 1.2041133
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2016169, upper bound: 1.2037857
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2037870, upper bound: 1.2016156
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2041135, upper bound: 1.2012694
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2013406, upper bound: 1.2040419
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.54
Output dim: 6, lower bound: -1.2016882, upper bound: 1.2037145

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2200465, 3.2197819
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3269229, 3.3236687
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5779800, 3.5808077
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8929319, 3.8938646
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5888052, 3.5908480
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1301384, 3.1312690
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9905534, 2.9883976
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9853268, 2.9855914
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6462984, 3.6471510
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5000029, 2.5068889

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2010332, upper bound: 1.2039665
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2010332, upper bound: 1.2035619
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2194924, 3.2203355
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3262038, 3.3243849
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5781355, 3.5806518
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8931990, 3.8935966
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5888033, 3.5908499
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1301117, 3.1312952
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9907727, 2.9881768
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9848232, 2.9860935
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6467686, 3.6466799
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.4997473, 2.5071430

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2010342, upper bound: 1.2035614
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2014491, upper bound: 1.2035615
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2203355, 3.2194924
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3243852, 3.3262043
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5806522, 3.5781355
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8935966, 3.8931990
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5908499, 3.5888033
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1312943, 3.1301122
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9881768, 2.9907732
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9860935, 2.9848237
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6466799, 3.6467686
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5071421, 2.4997473

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2035618, upper bound: 1.2014476
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2035618, upper bound: 1.2010327
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2197824, 3.2200465
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3236690, 3.3269229
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5808077, 3.5779800
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8938646, 3.8929310
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5908480, 3.5888052
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1312695, 3.1301389
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9883981, 2.9905529
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9855919, 2.9853268
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6471510, 3.6462975
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5068884, 2.5000024

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2035626, upper bound: 1.2010321
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2039672, upper bound: 1.2010329
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2200227, 3.2198062
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3257060, 3.3248858
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5790577, 3.5797300
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8913202, 3.8954754
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5881243, 3.5915289
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1303425, 3.1310639
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9899926, 2.9889574
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9848652, 2.9860535
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6472101, 3.6462388
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.4992743, 2.5076175

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2011029, upper bound: 1.2038957
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2011029, upper bound: 1.2034920
time: 5.45 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.50 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2010332, upper bound: 1.2039665
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2010332, upper bound: 1.2035619
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2010342, upper bound: 1.2035614
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2014491, upper bound: 1.2035615
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2035618, upper bound: 1.2014476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2035618, upper bound: 1.2010327
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2035626, upper bound: 1.2010321
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2039672, upper bound: 1.2010329
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2011029, upper bound: 1.2038957
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.50
Output dim: 6, lower bound: -1.2011029, upper bound: 1.2034920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2202592, 3.2197795
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3255386, 3.3221045
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5779791, 3.5808949
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8929291, 3.8940363
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5884399, 3.5904355
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1298676, 3.1310282
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9899964, 2.9879045
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9838257, 2.9838958
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6462955, 3.6472797
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.4995151, 2.5063372

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.1992522, upper bound: 1.2023725
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.1994450, upper bound: 1.2021813
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2197795, 3.2202592
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3221045, 3.3255389
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5808954, 3.5779786
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8940363, 3.8929291
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5904350, 3.5884399
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1310282, 3.1298676
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9879050, 2.9899960
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9838963, 2.9838257
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6472797, 3.6462955
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5063376, 2.4995143

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2021816, upper bound: 1.1994442
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2023729, upper bound: 1.1992508
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2202353, 3.2198033
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.3243217, 3.3233216
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.5790567, 3.5798173
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.8913183, 3.8956470
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.5877590, 3.5911160
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1300716, 3.1308236
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9894357, 2.9884644
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9833641, 2.9843578
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.6472082, 3.6463671
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.4987864, 2.5070658

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.1993218, upper bound: 1.2023014
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.1995145, upper bound: 1.2021103
time: 4.86 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.41 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.41
Output dim: 6, lower bound: -1.1992522, upper bound: 1.2023725
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.41
Output dim: 6, lower bound: -1.1994450, upper bound: 1.2021813
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.41
Output dim: 6, lower bound: -1.2021816, upper bound: 1.1994442
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.41
Output dim: 6, lower bound: -1.2023729, upper bound: 1.1992508
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.41
Output dim: 6, lower bound: -1.1993218, upper bound: 1.2023014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.41
Output dim: 6, lower bound: -1.1995145, upper bound: 1.2021103
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.956814765930176
rel_dist={6: [-1.2041204236983045, 1.204119556210955]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3423995, upper bound: 1.3425422
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3425424, upper bound: 1.3423992
time: 5.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.96
Output dim: 6, lower bound: -1.3423995, upper bound: 1.3425422
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.96
Output dim: 6, lower bound: -1.3425424, upper bound: 1.3423992

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3312874, 3.3312387
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6155558, 3.6131215
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7750101, 3.7771635
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0754986, 4.0722775
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7518520, 3.7504902
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2086377, 3.2090468
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0313473, 3.0302281
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0995350, 3.0986118
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7631006, 3.7649264
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7212496, 2.7197928

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3423919, upper bound: 1.3406183
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3404757, upper bound: 1.3425345
time: 4.91 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3312387, 3.3312869
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6131220, 3.6155558
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7771635, 3.7750087
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0722780, 4.0754991
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7504902, 3.7518520
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2090478, 3.2086377
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0302277, 3.0313478
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0986118, 3.0995350
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7649260, 3.7631011
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7197924, 2.7212503

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3425348, upper bound: 1.3404751
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3406186, upper bound: 1.3423915
time: 5.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 6, lower bound: -1.3423919, upper bound: 1.3406183
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 6, lower bound: -1.3404757, upper bound: 1.3425345
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 6, lower bound: -1.3425348, upper bound: 1.3404751
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 6, lower bound: -1.3406186, upper bound: 1.3423915

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3302040, 3.3304009
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5907774, 3.5938778
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7571421, 3.7632875
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0674343, 4.0618706
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7495832, 3.7475696
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2071028, 3.2070694
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0307035, 3.0293999
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0986233, 3.0974226
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7508144, 3.7490935
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7062483, 2.7004764

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3423913, upper bound: 1.3398075
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3415808, upper bound: 1.3406174
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3304491, 3.3301563
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5963125, 3.5883422
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7611341, 3.7592959
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0650921, 4.0642114
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7489319, 3.7482209
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2066603, 3.2075133
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0305195, 3.0295839
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0983467, 3.0977001
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7472687, 3.7526388
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7019339, 2.7047911

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3404751, upper bound: 1.3417233
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3396651, upper bound: 1.3425339
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3301563, 3.3304491
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5883417, 3.5963125
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7592974, 3.7611327
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0642128, 4.0650921
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7482214, 3.7489309
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2075129, 3.2066598
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0295839, 3.0305195
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0977001, 3.0983462
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7526398, 3.7472682
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7047911, 2.7019339

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3425342, upper bound: 1.3396645
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417237, upper bound: 1.3404744
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3304005, 3.3302040
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5938787, 3.5907764
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7632875, 3.7571416
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0618706, 4.0674329
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7475700, 3.7495823
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2070684, 3.2071042
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0293999, 3.0307035
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0974236, 3.0986238
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7490940, 3.7508140
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7004766, 2.7062485

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3406180, upper bound: 1.3415804
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3398081, upper bound: 1.3423910
time: 5.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3423913, upper bound: 1.3398075
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3415808, upper bound: 1.3406174
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3404751, upper bound: 1.3417233
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3396651, upper bound: 1.3425339
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3425342, upper bound: 1.3396645
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3417237, upper bound: 1.3404744
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3406180, upper bound: 1.3415804
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.72
Output dim: 6, lower bound: -1.3398081, upper bound: 1.3423910

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3058815, 3.3032651
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4909296, 3.4848523
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6644945, 3.6622143
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0538292, 4.0499606
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7452784, 3.7436218
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2030210, 3.2033267
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0399585, 3.0395937
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1113081, 3.1071491
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7357063, 3.7361536
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6864576, 2.6775692

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3423889, upper bound: 1.3376356
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3402194, upper bound: 1.3398053
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3030682, 3.3060780
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4817514, 3.4940305
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6560678, 3.6706409
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0555229, 4.0482664
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7456350, 3.7432656
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2033615, 3.2029867
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0408969, 3.0386558
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1083498, 3.1101079
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7378740, 3.7339854
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6833410, 2.6806855

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3415783, upper bound: 1.3384458
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3394089, upper bound: 1.3406151
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3061256, 3.3030200
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4964647, 3.4793162
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6684866, 3.6582227
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0514889, 4.0523019
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7446270, 3.7442732
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2025766, 3.2037706
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0397754, 3.0397778
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1110315, 3.1074266
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7321606, 3.7396994
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6821432, 2.6818838

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3404726, upper bound: 1.3395517
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3383036, upper bound: 1.3417212
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3033133, 3.3058333
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4872866, 3.4884944
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6600599, 3.6666498
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0531826, 4.0506072
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7449827, 3.7439170
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2029171, 3.2034302
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0407138, 3.0388393
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1080713, 3.1103854
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7343283, 3.7375307
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6790266, 2.6850002

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3396627, upper bound: 1.3403620
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3374934, upper bound: 1.3425308
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3058329, 3.3033128
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4884949, 3.4872866
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6666498, 3.6600590
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0506077, 4.0531821
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7439165, 3.7449832
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2034302, 3.2029171
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0388398, 3.0407133
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1103849, 3.1080728
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7375307, 3.7343283
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6850004, 2.6790266

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3425318, upper bound: 1.3374927
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403624, upper bound: 1.3396624
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3030205, 3.3061261
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4793167, 3.4964647
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6582232, 3.6684866
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0523014, 4.0514879
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7442732, 3.7446270
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2037706, 3.2025771
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0397773, 3.0397749
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1074266, 3.1110311
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7396994, 3.7321606
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6818838, 2.6821427

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417213, upper bound: 1.3383029
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3395518, upper bound: 1.3404723
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3060780, 3.3030682
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4940310, 3.4817505
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6706419, 3.6560678
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0482674, 4.0555234
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7432661, 3.7456346
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2029867, 3.2033615
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0386558, 3.0408969
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1101084, 3.1083503
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7339859, 3.7378740
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6806860, 2.6833413

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3406156, upper bound: 1.3394086
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3384465, upper bound: 1.3415781
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3032646, 3.3058810
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4848528, 3.4909286
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6622152, 3.6644950
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0499611, 4.0538287
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7436218, 3.7452784
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2033272, 3.2030210
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0395942, 3.0399590
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1071482, 3.1113086
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7361536, 3.7357059
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6775694, 2.6864576

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3398056, upper bound: 1.3402190
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3376363, upper bound: 1.3423884
time: 4.85 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3423889, upper bound: 1.3376356
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3402194, upper bound: 1.3398053
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3415783, upper bound: 1.3384458
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3394089, upper bound: 1.3406151
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3404726, upper bound: 1.3395517
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3383036, upper bound: 1.3417212
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3396627, upper bound: 1.3403620
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3374934, upper bound: 1.3425308
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3425318, upper bound: 1.3374927
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3403624, upper bound: 1.3396624
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3417213, upper bound: 1.3383029
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3395518, upper bound: 1.3404723
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3406156, upper bound: 1.3394086
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3384465, upper bound: 1.3415781
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3398056, upper bound: 1.3402190
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 6, lower bound: -1.3376363, upper bound: 1.3423884

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3007369, 3.3000622
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4486384, 3.4488409
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6508961, 3.6498609
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0108757, 4.0031023
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6885443, 3.6817331
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1876345, 3.1861405
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1049614, 3.1015801
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7064342, 3.7093186
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5949988, 2.5778008

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3416450, upper bound: 1.3376348
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3423882, upper bound: 1.3368923
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3026776, 3.2981210
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4549174, 3.4425619
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6521425, 3.6486144
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0069714, 4.0070071
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6833897, 3.6868877
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1858339, 3.1879401
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1057396, 3.1008029
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7088718, 3.7068806
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5866895, 2.5861099

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3394755, upper bound: 1.3398041
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3402187, upper bound: 1.3390618
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2979236, 3.3028750
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4394603, 3.4580188
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6424685, 3.6582880
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0125694, 4.0014081
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6889009, 3.6813769
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1879740, 3.1858001
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1020031, 3.1045389
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7086010, 3.7071500
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5918822, 2.5809169

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3408345, upper bound: 1.3384444
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3415776, upper bound: 1.3377029
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2998652, 3.3009338
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4457393, 3.4517400
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6437149, 3.6570420
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0086651, 4.0053134
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6837454, 3.6865320
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1861753, 3.1876001
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1027813, 3.1037617
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7110386, 3.7047124
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5835729, 2.5892262

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3386648, upper bound: 1.3406134
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3394082, upper bound: 1.3398724
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3009820, 3.2998171
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4541736, 3.4433050
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6548872, 3.6458697
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0085335, 4.0054436
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6878939, 3.6823845
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1871901, 3.1865845
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1046848, 3.1018577
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7028885, 3.7128634
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5906835, 2.5821154

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3397301, upper bound: 1.3395501
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3404719, upper bound: 1.3388080
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3029227, 3.2978764
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4604526, 3.4370260
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6561337, 3.6446233
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0046291, 4.0093489
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6827383, 3.6875391
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1853914, 3.1883845
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1054611, 3.1010804
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7053261, 3.7104259
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5823741, 2.5904245

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3375605, upper bound: 1.3417195
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3383029, upper bound: 1.3409777
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2981687, 3.3026299
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4449954, 3.4524829
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6464605, 3.6542964
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0102291, 4.0037494
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6882496, 3.6820283
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1875315, 3.1862440
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1017265, 3.1048160
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7050552, 3.7106957
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5875678, 2.5852318

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3389196, upper bound: 1.3403602
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3396620, upper bound: 1.3396183
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3001094, 3.3006887
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4512744, 3.4462039
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6477060, 3.6530504
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0063229, 4.0076547
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6830950, 3.6871834
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1857309, 3.1880441
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1025028, 3.1040392
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7074947, 3.7082582
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5792584, 2.5935409

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3367500, upper bound: 1.3425300
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3374927, upper bound: 1.3417878
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3006892, 3.3001099
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4462047, 3.4512751
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6530504, 3.6477060
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0076542, 4.0063238
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6871834, 3.6830945
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1880445, 3.1857314
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1040382, 3.1025033
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7082577, 3.7074933
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5935407, 2.5792582

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417884, upper bound: 1.3374914
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3425311, upper bound: 1.3367493
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3026299, 3.2981691
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4524837, 3.4449961
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6542969, 3.6464601
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0037498, 4.0102286
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6820288, 3.6882496
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1862440, 3.1875310
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1048164, 3.1017261
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7106953, 3.7050557
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5852313, 2.5875673

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3396187, upper bound: 1.3396607
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403617, upper bound: 1.3389184
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2978759, 3.3029232
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4370265, 3.4604533
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6446238, 3.6561332
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0093479, 4.0046296
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6875391, 3.6827383
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1883841, 3.1853909
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1010799, 3.1054626
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7104263, 3.7053251
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5904250, 2.5823743

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3409776, upper bound: 1.3383014
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417206, upper bound: 1.3375594
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2998176, 3.3009820
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4433055, 3.4541743
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6458702, 3.6548867
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0054436, 4.0085344
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6823845, 3.6878934
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1865835, 3.1871905
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1018562, 3.1046853
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7128639, 3.7028871
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5821157, 2.5906835

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3388079, upper bound: 1.3404705
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3395511, upper bound: 1.3397289
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3009334, 3.2998652
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4517398, 3.4457393
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6570425, 3.6437144
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0053139, 4.0086646
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6865320, 3.6837459
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1876001, 3.1861749
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1037617, 3.1027808
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7047138, 3.7110386
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5892262, 2.5835729

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3398735, upper bound: 1.3394072
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3406149, upper bound: 1.3386649
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3028750, 3.2979240
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4580188, 3.4394603
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6582880, 3.6424685
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0014076, 4.0125699
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6813765, 3.6889009
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1857996, 3.1879749
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1045380, 3.1020036
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7071514, 3.7086010
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5809169, 2.5918820

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3377039, upper bound: 1.3415766
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3384458, upper bound: 1.3408344
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2981210, 3.3026781
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4425616, 3.4549172
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6486149, 3.6521420
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0070076, 4.0069709
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6868877, 3.6833897
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1879396, 3.1858344
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1008034, 3.1057396
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7068806, 3.7088704
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5861096, 2.5866892

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3390629, upper bound: 1.3402174
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3398049, upper bound: 1.3394754
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3000617, 3.3007374
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4488406, 3.4486382
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6498613, 3.6508956
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0031013, 4.0108762
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6817331, 3.6885448
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1861410, 3.1876345
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1015797, 3.1049628
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7093182, 3.7064328
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5778003, 2.5949984

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3368932, upper bound: 1.3423868
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3376356, upper bound: 1.3416444
time: 4.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3416450, upper bound: 1.3376348
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3423882, upper bound: 1.3368923
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3394755, upper bound: 1.3398041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3402187, upper bound: 1.3390618
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3408345, upper bound: 1.3384444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3415776, upper bound: 1.3377029
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3386648, upper bound: 1.3406134
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3394082, upper bound: 1.3398724
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3397301, upper bound: 1.3395501
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3404719, upper bound: 1.3388080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3375605, upper bound: 1.3417195
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3383029, upper bound: 1.3409777
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3389196, upper bound: 1.3403602
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3396620, upper bound: 1.3396183
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3367500, upper bound: 1.3425300
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3374927, upper bound: 1.3417878
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3417884, upper bound: 1.3374914
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3425311, upper bound: 1.3367493
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3396187, upper bound: 1.3396607
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3403617, upper bound: 1.3389184
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3409776, upper bound: 1.3383014
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3417206, upper bound: 1.3375594
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3388079, upper bound: 1.3404705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3395511, upper bound: 1.3397289
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3398735, upper bound: 1.3394072
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3406149, upper bound: 1.3386649
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3377039, upper bound: 1.3415766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3384458, upper bound: 1.3408344
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3390629, upper bound: 1.3402174
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3398049, upper bound: 1.3394754
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3368932, upper bound: 1.3423868
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.59
Output dim: 6, lower bound: -1.3376356, upper bound: 1.3416444
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.0204544067382812
rel_dist={6: [-1.3425452667053426, 1.3425424335470915]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 3038.77 seconds
