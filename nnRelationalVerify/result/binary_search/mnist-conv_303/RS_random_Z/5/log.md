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
execution time: IAR + LP analysis = 14.97 + 32.37 = 47.34 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.66 seconds, max iter: 100)

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
Binary search time: 152.09 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 3400.57 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8405111, upper bound: 1.8394877
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8394881, upper bound: 1.8405107
time: 5.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.96
Output dim: 6, lower bound: -1.8405111, upper bound: 1.8394877
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.96
Output dim: 6, lower bound: -1.8394881, upper bound: 1.8405107

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6357989, 3.6273599
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9635949, 3.9360614
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4824228, 4.4875059
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4238887, 3.4249096
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5272532, 3.5183773
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9771938, 3.9836974
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.0096693, 3.0003209

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 877

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8405033, upper bound: 1.8322825
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8333080, upper bound: 1.8394800
time: 4.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6273599, 3.6357985
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9360623, 3.9635954
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4875059, 4.4824228
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4249101, 3.4238882
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5183764, 3.5272541
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9836979, 3.9771929
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -3.0003214, 3.0096698

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8394713, upper bound: 1.8351062
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8340849, upper bound: 1.8404939
time: 7.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.17 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 6, lower bound: -1.8405033, upper bound: 1.8322825
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 6, lower bound: -1.8333080, upper bound: 1.8394800
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 6, lower bound: -1.8394713, upper bound: 1.8351062
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 6, lower bound: -1.8340849, upper bound: 1.8404939

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6306539, 3.6280384
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9213042, 3.9126067
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4472790, 4.4406471
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4121013, 3.4077239
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5209064, 3.5143619
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9479208, 3.9617386
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9348288, 2.9005525

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8404864, upper bound: 1.8268707
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8350988, upper bound: 1.8322660
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6364770, 3.6222153
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9401412, 3.8937697
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4355640, 4.4523621
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4067035, 3.4131231
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5232382, 3.5120311
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9552345, 3.9544253
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9099016, 2.9254799

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330738, upper bound: 1.8394792
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8333070, upper bound: 1.8392025
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6262774, 3.6354513
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9112811, 3.9554229
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4841232, 4.4720163
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4242640, 3.4219108
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5180216, 3.5260653
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9785023, 3.9613609
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9939489, 2.9903529

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8322263, upper bound: 1.8281059
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8324787, upper bound: 1.8278535
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6270118, 3.6347165
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9278893, 3.9388151
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4770985, 4.4790397
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4229307, 3.4232421
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5171890, 3.5268970
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9678659, 3.9719973
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9810038, 3.0032971

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8339040, upper bound: 1.8367257
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8303275, upper bound: 1.8403072
time: 4.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8404864, upper bound: 1.8268707
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8350988, upper bound: 1.8322660
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8330738, upper bound: 1.8394792
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8333070, upper bound: 1.8392025
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8322263, upper bound: 1.8281059
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8324787, upper bound: 1.8278535
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8339040, upper bound: 1.8367257
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.25
Output dim: 6, lower bound: -1.8303275, upper bound: 1.8403072

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6295724, 3.6276913
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8965249, 3.9044352
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4438963, 4.4302397
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4114556, 3.4057460
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5205507, 3.5131731
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9427252, 3.9459066
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9284563, 2.8812363

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8332397, upper bound: 1.8198807
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8334923, upper bound: 1.8196281
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6303067, 3.6269569
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9131322, 3.8878274
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4368715, 4.4372630
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4101243, 3.4070778
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5197182, 3.5140052
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9320889, 3.9565430
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9155121, 2.8941805

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8349183, upper bound: 1.8321105
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8277736, upper bound: 1.8321208
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6370816, 3.6226759
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9181905, 3.8645163
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4563408, 4.4634738
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4106350, 3.4182839
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5348930, 3.5209155
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9543734, 3.9590392
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9014230, 2.9126291

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8328753, upper bound: 1.8321448
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8328643, upper bound: 1.8392923
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6369376, 3.6228199
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9108872, 3.8718190
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4466763, 4.4731374
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4118633, 3.4170556
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5321236, 3.5236859
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9598475, 3.9535632
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8970504, 2.9170012

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8320304, upper bound: 1.8379701
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8310130, upper bound: 1.8379701
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6298366, 3.6352797
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9174271, 3.9551253
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4849138, 4.4719777
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4241343, 3.4245720
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5196619, 3.5259852
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9784145, 3.9631333
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9966178, 2.9902225

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8320555, upper bound: 1.8243573
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8284723, upper bound: 1.8279464
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6261058, 3.6354513
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9109840, 3.9554229
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4840841, 4.4720163
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4242640, 3.4217806
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5179415, 3.5260653
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9785023, 3.9612732
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9938178, 2.9903529

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8324709, upper bound: 1.8206485
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8252720, upper bound: 1.8278454
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6270118, 3.6347256
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9278750, 3.9388089
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4770985, 4.4790354
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4229317, 3.4232397
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5171833, 3.5268965
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9678621, 3.9719934
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9810038, 3.0032969

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8266752, upper bound: 1.8297453
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8269273, upper bound: 1.8294927
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6270213, 3.6347160
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9278827, 3.9388008
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4770947, 4.4790382
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4229298, 3.4232421
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5171871, 3.5268922
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9678612, 3.9719944
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9810038, 3.0032964

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8267505, upper bound: 1.8331434
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8267164, upper bound: 1.8402990
time: 4.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8332397, upper bound: 1.8198807
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8334923, upper bound: 1.8196281
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8349183, upper bound: 1.8321105
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8277736, upper bound: 1.8321208
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8328753, upper bound: 1.8321448
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8328643, upper bound: 1.8392923
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8320304, upper bound: 1.8379701
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8310130, upper bound: 1.8379701
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8320555, upper bound: 1.8243573
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8284723, upper bound: 1.8279464
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8324709, upper bound: 1.8206485
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8252720, upper bound: 1.8278454
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8266752, upper bound: 1.8297453
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8269273, upper bound: 1.8294927
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8267505, upper bound: 1.8331434
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.28
Output dim: 6, lower bound: -1.8267164, upper bound: 1.8402990

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6331301, 3.6275191
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9026709, 3.9041371
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4446878, 4.4302020
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4113269, 3.4084082
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5221910, 3.5130930
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9426374, 3.9476786
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9311256, 2.8811057

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330696, upper bound: 1.8198626
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330701, upper bound: 1.8194879
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6294003, 3.6276913
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8962259, 3.9044352
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4438581, 4.4302397
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4114556, 3.4056168
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5204706, 3.5131731
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9427252, 3.9458189
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9283257, 2.8812363

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8317371, upper bound: 1.8192054
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8330681, upper bound: 1.8178749
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6303077, 3.6269660
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9131312, 3.8878379
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4368715, 4.4372597
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4101262, 3.4070778
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5197182, 3.5140100
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9320869, 3.9565420
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9155121, 2.8941801

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8345948, upper bound: 1.8321095
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8349174, upper bound: 1.8318452
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6303163, 3.6269569
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9131408, 3.8878264
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4368696, 4.4372625
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4101243, 3.4070802
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5197229, 3.5140057
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9320850, 3.9565411
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9155121, 2.8941796

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8260201, upper bound: 1.8316943
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8273494, upper bound: 1.8303657
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6370835, 3.6226864
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9181914, 3.8645251
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4563389, 4.4634700
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4106369, 3.4182830
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5348930, 3.5209198
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9543715, 3.9590354
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9014220, 2.9126287

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8256615, upper bound: 1.8251459
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8259140, upper bound: 1.8248935
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6370921, 3.6226768
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9182029, 3.8645172
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4563370, 4.4634728
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4106350, 3.4182849
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5348978, 3.5209160
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9543715, 3.9590373
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9014220, 2.9126291

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8285406, upper bound: 1.8392776
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8328497, upper bound: 1.8349840
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6365681, 3.6226468
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9107080, 3.8717380
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4472904, 4.4742332
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4116650, 3.4169040
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5311012, 3.5229263
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9598989, 3.9539652
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8981419, 2.9177341

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8247817, upper bound: 1.8309858
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8250340, upper bound: 1.8307333
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6363678, 3.6224499
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9106107, 3.8716393
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4477730, 4.4737520
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4117126, 3.4168563
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5308552, 3.5226641
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9602518, 3.9536138
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8977833, 2.9180913

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8308640, upper bound: 1.8306347
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8308526, upper bound: 1.8377655
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6298356, 3.6352882
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9174128, 3.9551187
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4849110, 4.4719729
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4241333, 3.4245701
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5196571, 3.5259848
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9784126, 3.9631300
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9966159, 2.9902215

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8320477, upper bound: 1.8207404
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248696, upper bound: 1.8207745
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6298451, 3.6352792
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9174204, 3.9551105
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4849091, 4.4719758
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4241314, 3.4245720
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5196609, 3.5259805
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9784107, 3.9631310
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9966168, 2.9902210

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248797, upper bound: 1.8207495
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8248595, upper bound: 1.8279384
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6209612, 3.6361303
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8686914, 3.9319692
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4489412, 4.4251566
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4124780, 3.4045959
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5115938, 3.5220494
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9492302, 3.9393139
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9189768, 2.8905852

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8311756, upper bound: 1.8183533
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8311754, upper bound: 1.8193713
time: 5.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6267843, 3.6303072
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8875284, 3.9131322
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4372263, 4.4368715
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4070783, 3.4099951
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5139256, 3.5197186
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9565430, 3.9320006
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8940496, 2.9155126

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8232700, upper bound: 1.8278433
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8252703, upper bound: 1.8258451
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6305699, 3.6345539
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9340200, 3.9385109
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4778881, 4.4789963
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4228020, 3.4259014
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5188255, 3.5268168
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9677763, 3.9737663
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9836717, 3.0031657

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8265053, upper bound: 1.8297270
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8265057, upper bound: 1.8293523
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6268401, 3.6347256
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9275770, 3.9388089
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4770584, 4.4790354
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4229317, 3.4231100
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5171032, 3.5268965
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9678621, 3.9719062
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9808726, 3.0032969

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8256356, upper bound: 1.8272007
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8256355, upper bound: 1.8282194
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6218772, 3.6353955
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.8856063, 3.9153609
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4419508, 4.4321795
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4111466, 3.4060593
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5108461, 3.5228820
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9385900, 3.9500360
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.9061623, 2.9035285

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8247521, upper bound: 1.8331418
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8267488, upper bound: 1.8311431
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.6277003, 3.6295729
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.9044452, 3.8965240
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.4302378, 4.4438953
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.9909687, 3.9909687
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.4057469, 3.4114585
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.5131779, 3.5205512
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.9459066, 3.9427233
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8812351, 2.9284568

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 946

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8254200, upper bound: 1.8380027
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.8254199, upper bound: 1.8390222
time: 4.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8330696, upper bound: 1.8198626
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8330701, upper bound: 1.8194879
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8317371, upper bound: 1.8192054
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8330681, upper bound: 1.8178749
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8345948, upper bound: 1.8321095
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8349174, upper bound: 1.8318452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8260201, upper bound: 1.8316943
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8273494, upper bound: 1.8303657
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8256615, upper bound: 1.8251459
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8259140, upper bound: 1.8248935
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8285406, upper bound: 1.8392776
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8328497, upper bound: 1.8349840
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8247817, upper bound: 1.8309858
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8250340, upper bound: 1.8307333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8308640, upper bound: 1.8306347
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8308526, upper bound: 1.8377655
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8320477, upper bound: 1.8207404
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8248696, upper bound: 1.8207745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8248797, upper bound: 1.8207495
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8248595, upper bound: 1.8279384
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8311756, upper bound: 1.8183533
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8311754, upper bound: 1.8193713
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8232700, upper bound: 1.8278433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8252703, upper bound: 1.8258451
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8265053, upper bound: 1.8297270
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8265057, upper bound: 1.8293523
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8256356, upper bound: 1.8272007
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8256355, upper bound: 1.8282194
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8247521, upper bound: 1.8331418
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8267488, upper bound: 1.8311431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8254200, upper bound: 1.8380027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.15
Output dim: 6, lower bound: -1.8254199, upper bound: 1.8390222
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.050609588623047
rel_dist={6: [-1.840513682283781, 1.840511023435588]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4737506, upper bound: 1.4739430
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4739433, upper bound: 1.4737505
time: 8.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.56
Output dim: 6, lower bound: -1.4737506, upper bound: 1.4739430
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.56
Output dim: 6, lower bound: -1.4739433, upper bound: 1.4737505

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4134002, 3.4115353
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7459764, 3.7427549
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1702576, 4.1698427
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8608036, 3.8602228
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2603931, 3.2617893
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1952400, 3.1943798
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8209591, 3.8218889
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8065577, 2.8051589

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4735215, upper bound: 1.4739421
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4737496, upper bound: 1.4737143
time: 7.24 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4115348, 3.4117079
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7427549, 3.7430525
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1698427, 4.1698799
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8602228, 3.8602772
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2605228, 3.2603931
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1943798, 3.1944599
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8210468, 3.8209591
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8051596, 2.8052888

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4726144, upper bound: 1.4731355
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4733287, upper bound: 1.4724213
time: 6.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 28.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.39
Output dim: 6, lower bound: -1.4735215, upper bound: 1.4739421
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.39
Output dim: 6, lower bound: -1.4737496, upper bound: 1.4737143
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.39
Output dim: 6, lower bound: -1.4726144, upper bound: 1.4731355
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.39
Output dim: 6, lower bound: -1.4733287, upper bound: 1.4724213

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4139333, 3.4119964
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7203741, 3.7135010
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1862011, 4.1809545
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8464308, 3.8438082
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2643256, 3.2663355
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2055082, 3.2032633
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8200998, 3.8237681
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7958937, 2.7923079

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721919, upper bound: 1.4733273
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729062, upper bound: 1.4726130
time: 8.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4138608, 3.4120684
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7167225, 3.7171526
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1813698, 4.1857862
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8443890, 3.8458505
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2649398, 3.2657213
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2041235, 3.2046480
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8228388, 3.8210301
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7937078, 2.7944939

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4737495, upper bound: 1.4720319
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4720661, upper bound: 1.4737141
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4121785, 3.4117055
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7417307, 3.7414904
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1698399, 4.1703982
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8599529, 3.8598661
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2602520, 3.2602148
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1932650, 3.1927633
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8210449, 3.8213491
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8047957, 2.8047371

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721167, upper bound: 1.4731345
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721202, upper bound: 1.4719257
time: 5.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4115329, 3.4123502
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7411909, 3.7420282
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1703606, 4.1698775
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8598127, 3.8600059
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2603445, 3.2601223
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1926832, 3.1933451
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8214359, 3.8209567
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8046069, 2.8049266

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729969, upper bound: 1.4714660
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723737, upper bound: 1.4720892
time: 6.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4721919, upper bound: 1.4733273
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4729062, upper bound: 1.4726130
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4737495, upper bound: 1.4720319
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4720661, upper bound: 1.4737141
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4721167, upper bound: 1.4731345
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4721202, upper bound: 1.4719257
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4729969, upper bound: 1.4714660
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.86
Output dim: 6, lower bound: -1.4723737, upper bound: 1.4720892

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4145761, 3.4119940
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7193489, 3.7119365
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1861973, 4.1814713
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8461599, 3.8433976
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2640548, 3.2661562
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2043943, 3.2015667
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8200979, 3.8241587
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7955322, 2.7917566

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721822, upper bound: 1.4704720
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4693384, upper bound: 1.4733178
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4139304, 3.4126387
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7188091, 3.7124748
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1867180, 4.1809506
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8460207, 3.8435373
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2641463, 3.2660642
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2038116, 3.2021494
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8204908, 3.8237658
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7953424, 2.7919462

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729024, upper bound: 1.4692140
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4695046, upper bound: 1.4726092
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4138618, 3.4120736
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7167091, 3.7171421
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1813660, 4.1857824
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8443890, 3.8458509
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2649388, 3.2657189
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2041178, 3.2046452
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8228350, 3.8210263
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7937078, 2.7944942

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4724077, upper bound: 1.4713934
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731219, upper bound: 1.4706792
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4138665, 3.4120688
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7167130, 3.7171378
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1813641, 4.1857839
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8443890, 3.8458509
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2649369, 3.2657199
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2041206, 3.2046428
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8228350, 3.8210273
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7937078, 2.7944939

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4708983, upper bound: 1.4737130
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4720652, upper bound: 1.4725410
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4152803, 3.4131432
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7266192, 3.7242212
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1676235, 4.1689887
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8598986, 3.8598061
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2606974, 3.2605820
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1826754, 3.1806622
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8192444, 3.8209629
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7994261, 2.7986007

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718624, upper bound: 1.4730572
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721093, upper bound: 1.4730619
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4136162, 3.4148054
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7244620, 3.7263765
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1684284, 4.1681833
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8598909, 3.8598118
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2606192, 3.2605028
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1811647, 3.1821709
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8206577, 3.8195486
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7986593, 2.7993658

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721105, upper bound: 1.4690676
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4692651, upper bound: 1.4719139
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4119568, 3.4128714
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7413983, 3.7422843
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1683865, 4.1681447
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8615017, 3.8613672
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2601447, 3.2599463
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1922226, 3.1930161
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8200607, 3.8197579
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8033552, 2.8034966

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729872, upper bound: 1.4686092
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4701444, upper bound: 1.4714563
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4120550, 3.4127731
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7414479, 3.7422352
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1686268, 4.1679034
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8611746, 3.8616943
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2601676, 3.2599225
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1923542, 3.1928854
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8202381, 3.8195815
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.8031759, 2.8036752

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4723640, upper bound: 1.4692353
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4695184, upper bound: 1.4720794
time: 7.49 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4721822, upper bound: 1.4704720
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4693384, upper bound: 1.4733178
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4729024, upper bound: 1.4692140
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4695046, upper bound: 1.4726092
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4724077, upper bound: 1.4713934
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4731219, upper bound: 1.4706792
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4708983, upper bound: 1.4737130
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4720652, upper bound: 1.4725410
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4718624, upper bound: 1.4730572
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4721093, upper bound: 1.4730619
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4721105, upper bound: 1.4690676
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4692651, upper bound: 1.4719139
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4729872, upper bound: 1.4686092
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4701444, upper bound: 1.4714563
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4723640, upper bound: 1.4692353
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.78
Output dim: 6, lower bound: -1.4695184, upper bound: 1.4720794

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4134927, 3.4112787
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6945686, 3.6954613
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1793041, 4.1710658
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8442154, 3.8404760
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2627430, 3.2641788
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2036228, 3.2003784
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8095837, 3.8083253
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7826872, 2.7724392

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721071, upper bound: 1.4704651
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721049, upper bound: 1.4702167
time: 5.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4138598, 3.4109111
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7028732, 3.6871572
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1757927, 4.1745772
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8432388, 3.8414536
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2620764, 3.2648449
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2032070, 3.2007947
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8042650, 3.8136435
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7762146, 2.7789114

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4693368, upper bound: 1.4720578
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4680753, upper bound: 1.4733170
time: 6.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4087873, 3.4104061
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6765213, 3.6796045
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1457195, 4.1340938
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7918644, 3.7816486
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2496605, 3.2488775
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1974669, 3.1969686
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7912178, 3.7981496
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7080383, 2.6921778

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4716903, upper bound: 1.4687214
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729015, upper bound: 1.4687181
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4116988, 3.4074945
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6859398, 3.6701860
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1398621, 4.1399512
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7841320, 3.7893810
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2469606, 3.2515774
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1986322, 3.1958032
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7948751, 3.7944927
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6955738, 2.7046416

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4682950, upper bound: 1.4721153
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4695038, upper bound: 1.4721111
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4145036, 3.4120708
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7156830, 3.7155776
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1813622, 4.1862993
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8441172, 3.8454390
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2646670, 3.2655401
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2030039, 3.2029486
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8228331, 3.8214169
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7933445, 2.7939417

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4724039, upper bound: 1.4696818
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4690083, upper bound: 1.4696902
time: 9.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4138589, 3.4127154
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7151442, 3.7161164
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1818829, 4.1857786
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8439770, 3.8455791
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2647595, 3.2654476
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.2024221, 3.2035308
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8232260, 3.8210244
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7931557, 2.7941313

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4719105, upper bound: 1.4701853
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731210, upper bound: 1.4701821
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4169664, 3.4135060
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7015991, 3.6998692
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1791515, 4.1843734
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8443336, 3.8457899
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2653828, 3.2660866
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1935291, 3.1925430
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8210344, 3.8206406
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7883358, 2.7883568

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4708970, upper bound: 1.4724527
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4696379, upper bound: 1.4737123
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4153032, 3.4151688
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6994438, 3.7020245
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1799545, 4.1835694
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8443279, 3.8457956
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2653036, 3.2661657
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1920204, 3.1940517
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8224478, 3.8192272
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7875710, 2.7891219

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4720642, upper bound: 1.4712808
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4708054, upper bound: 1.4725398
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4136696, 3.4123688
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7226954, 3.7160196
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1668139, 4.1672726
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8588333, 3.8575802
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2563086, 3.2584863
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1820841, 3.1794267
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8127403, 3.8178596
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7988567, 2.7974124

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4718586, upper bound: 1.4696544
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4684617, upper bound: 1.4730530
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4145060, 3.4115319
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7184172, 3.7202978
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1659184, 4.1681681
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8576727, 3.8587408
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2586031, 3.2561927
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1814394, 3.1800709
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8161335, 3.8144674
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7982368, 2.7980323

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4720996, upper bound: 1.4702028
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4692541, upper bound: 1.4730500
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4125347, 3.4140911
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6996818, 3.7099013
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1615334, 4.1577768
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8579473, 3.8568888
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2593079, 3.2585258
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1803923, 3.1809831
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8101454, 3.8037171
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7858152, 2.7800493

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721067, upper bound: 1.4656806
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4687132, upper bound: 1.4690635
time: 6.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4129019, 3.4137239
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7079864, 3.7015972
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1580219, 4.1612887
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8569698, 3.8578663
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2586403, 3.2591915
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1799765, 3.1813993
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8048267, 3.8090353
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7793436, 2.7865214

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689355, upper bound: 1.4709592
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4683092, upper bound: 1.4715821
time: 6.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4108763, 3.4121580
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7166181, 3.7258077
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1614914, 4.1577377
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8595581, 3.8584476
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2588334, 3.2579689
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1914511, 3.1918273
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8095455, 3.8039241
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7905111, 2.7841794

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729863, upper bound: 1.4679654
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4717218, upper bound: 1.4679966
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.4112434, 3.4117908
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.7249227, 3.7175040
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.1579790, 4.1612492
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.8585806, 3.8594246
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2581677, 3.2586350
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.1910353, 3.1922436
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.8042269, 3.8092422
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7840395, 2.7906516

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4689325, upper bound: 1.4709616
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4701435, upper bound: 1.4709612
time: 6.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4721071, upper bound: 1.4704651
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4721049, upper bound: 1.4702167
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4693368, upper bound: 1.4720578
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4680753, upper bound: 1.4733170
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4716903, upper bound: 1.4687214
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4729015, upper bound: 1.4687181
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4682950, upper bound: 1.4721153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4695038, upper bound: 1.4721111
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4724039, upper bound: 1.4696818
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4690083, upper bound: 1.4696902
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4719105, upper bound: 1.4701853
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4731210, upper bound: 1.4701821
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4708970, upper bound: 1.4724527
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4696379, upper bound: 1.4737123
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4720642, upper bound: 1.4712808
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4708054, upper bound: 1.4725398
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4718586, upper bound: 1.4696544
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4684617, upper bound: 1.4730530
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4720996, upper bound: 1.4702028
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4692541, upper bound: 1.4730500
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4721067, upper bound: 1.4656806
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4687132, upper bound: 1.4690635
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4689355, upper bound: 1.4709592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4683092, upper bound: 1.4715821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4729863, upper bound: 1.4679654
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4717218, upper bound: 1.4679966
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4689325, upper bound: 1.4709616
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.02
Output dim: 6, lower bound: -1.4701435, upper bound: 1.4709612
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 6, lower bound: -1.4723640, upper bound: 1.4692353
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.02
Output dim: 6, lower bound: -1.4695184, upper bound: 1.4720794
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.050609588623047
rel_dist={6: [-1.4762376626074394, 1.476239686090869]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2037936, upper bound: 1.2041222
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2041201, upper bound: 1.2037933
time: 7.14 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.47
Output dim: 6, lower bound: -1.2037936, upper bound: 1.2041222
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.47
Output dim: 6, lower bound: -1.2041201, upper bound: 1.2037933

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2518401, 3.2512851
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5251474, 3.5244286
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7338166, 3.7339725
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9502354, 3.9505038
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6734734, 3.6734715
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1492825, 3.1492562
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9515176, 2.9517388
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9734001, 2.9728971
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7050743, 3.7055454
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6541152, 2.6538601

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.2035684, upper bound: 1.2039722
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2035684, upper bound: 1.2035675
time: 6.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2512851, 3.2518396
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5244284, 3.5251472
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7339730, 3.7338166
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9505043, 3.9502358
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6734715, 3.6734734
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1492558, 3.1492825
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9517388, 2.9515185
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9728966, 2.9734001
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7055454, 3.7050738
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6538606, 2.6541152

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 877

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2028650, upper bound: 1.2028607
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2028650, upper bound: 1.2034352
time: 4.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.60
Output dim: 6, lower bound: -1.2035684, upper bound: 1.2039722
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 24.60
Output dim: 6, lower bound: -1.2035684, upper bound: 1.2035675
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 24.60
Output dim: 6, lower bound: -1.2028650, upper bound: 1.2028607
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 24.60
Output dim: 6, lower bound: -1.2028650, upper bound: 1.2034352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.2520537, 3.2512836
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5237646, 3.5228655
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7338157, 3.7340589
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9502316, 3.9506741
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6731110, 3.6730614
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1490107, 3.1490159
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -2.9509625, 2.9512467
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -2.9718981, 2.9712009
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7050743, 3.7056761
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6536274, 2.6533084

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2017957, upper bound: 1.2023814
time: 8.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.2019863, upper bound: 1.2021881
time: 5.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.32
Output dim: 6, lower bound: -1.2017957, upper bound: 1.2023814
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 28.32
Output dim: 6, lower bound: -1.2019863, upper bound: 1.2021881
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.956814765930176
rel_dist={6: [-1.2041204236983045, 1.204119556210955]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5860
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5860

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417997, upper bound: 1.3425445
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3425423, upper bound: 1.3417992
time: 4.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.55
Output dim: 6, lower bound: -1.3417997, upper bound: 1.3425445
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.55
Output dim: 6, lower bound: -1.3425423, upper bound: 1.3417992

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3333240, 3.3322148
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6265440, 3.6251063
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0589504, 4.0594869
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7668476, 3.7668433
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2051258, 3.2050729
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0151572, 3.0155988
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0786333, 3.0776277
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7621593, 3.7631016
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7270160, 2.7265062

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 946

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417990, upper bound: 1.3417334
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3409883, upper bound: 1.3425407
time: 5.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3322148, 3.3333235
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6251059, 3.6265435
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0594883, 4.0589509
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7668438, 3.7668471
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2050724, 3.2051258
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0155988, 3.0151572
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0776272, 3.0786333
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7631016, 3.7621589
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7265067, 2.7270162

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403912, upper bound: 1.3398377
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3405841, upper bound: 1.3396449
time: 5.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.84
Output dim: 6, lower bound: -1.3417990, upper bound: 1.3417334
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.84
Output dim: 6, lower bound: -1.3409883, upper bound: 1.3425407
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.84
Output dim: 6, lower bound: -1.3403912, upper bound: 1.3398377
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.84
Output dim: 6, lower bound: -1.3405841, upper bound: 1.3396449

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3089890, 3.3050699
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5266924, 3.5160818
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7077737, 3.6996565
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0453415, 4.0475717
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7625437, 3.7628956
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2010422, 3.2013297
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0244141, 3.0257936
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0913191, 3.0873551
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7470446, 3.7501545
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7072248, 2.7036006

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417966, upper bound: 1.3395580
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3396268, upper bound: 1.3417273
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3061776, 3.3078828
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5175200, 3.5252600
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6993470, 3.7080836
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0470371, 4.0458779
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7629004, 3.7625399
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2013836, 3.2009897
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0253525, 3.0248556
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0883608, 3.0903139
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7492132, 3.7479868
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7041101, 2.7067170

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 877

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3388318, upper bound: 1.3405832
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3390245, upper bound: 1.3403901
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3332863, 3.3331513
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6269569, 3.6262465
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0597248, 4.0589123
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7671766, 3.7667933
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2049446, 3.2059278
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0154843, 3.0158625
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0781212, 3.0785542
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7630167, 3.7626934
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7273088, 2.7268858

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403848, upper bound: 1.3387456
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3392985, upper bound: 1.3398291
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3320427, 3.3333235
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6248093, 3.6265435
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0594492, 4.0589509
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7667894, 3.7668471
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2050724, 3.2049975
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0155988, 3.0150433
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0775480, 3.0786333
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7631016, 3.7620735
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7263761, 2.7270162

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403212, upper bound: 1.3395662
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3405764, upper bound: 1.3395673
time: 5.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.54 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3417966, upper bound: 1.3395580
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3396268, upper bound: 1.3417273
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3388318, upper bound: 1.3405832
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3390245, upper bound: 1.3403901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3403848, upper bound: 1.3387456
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3392985, upper bound: 1.3398291
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3403212, upper bound: 1.3395662
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.54
Output dim: 6, lower bound: -1.3405764, upper bound: 1.3395673

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3038454, 3.3018665
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4844017, 3.4800701
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6941748, 3.6873045
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0023880, 4.0007129
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7058096, 3.7010069
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1856546, 3.1841426
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0427771, 3.0470347
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0849733, 3.0817871
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7177725, 3.7233200
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6157660, 2.6038320

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3416526, upper bound: 1.3395577
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417960, upper bound: 1.3394144
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3057861, 3.2999258
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4906807, 3.4737911
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6954212, 3.6860580
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9984837, 4.0046177
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7006550, 3.7061620
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1838541, 3.1859422
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0456553, 3.0441561
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0857496, 3.0810099
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7202101, 3.7208824
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6074557, 2.6121411

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3396193, upper bound: 1.3398043
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3377045, upper bound: 1.3417199
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3072495, 3.3077102
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5193691, 3.5249617
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6993260, 3.7082129
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0472746, 4.0458393
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7632332, 3.7624865
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2012520, 3.2017894
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0252380, 3.0255604
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0888548, 3.0902343
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7491255, 3.7485199
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7049131, 2.7065859

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3388247, upper bound: 1.3394907
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3377400, upper bound: 1.3405762
time: 9.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3060060, 3.3078828
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5172215, 3.5252600
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6993470, 3.7080626
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0469980, 4.0458779
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7628469, 3.7625399
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2013836, 3.2008591
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0253525, 3.0247412
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0882807, 3.0903139
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7492132, 3.7479000
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7039785, 2.7067170

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3386650, upper bound: 1.3400743
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3386650, upper bound: 1.3392903
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3332863, 3.3331542
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6269417, 3.6262341
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0597219, 4.0589085
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7671757, 3.7667923
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2049417, 3.2059250
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0154834, 3.0158606
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0781164, 3.0785499
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7630129, 3.7626901
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7273088, 2.7268853

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403824, upper bound: 1.3376575
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3382154, upper bound: 1.3376624
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3332891, 3.3331513
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6269455, 3.6262312
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0597210, 4.0589094
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7671757, 3.7667923
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2049417, 3.2059255
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0154834, 3.0158606
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0781174, 3.0785484
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7630129, 3.7626905
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7273088, 2.7268851

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3381998, upper bound: 1.3394818
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3389836, upper bound: 1.3394819
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3304319, 3.3322701
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6194592, 3.6183414
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0583410, 4.0572343
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7653389, 3.7646222
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2006826, 3.2021351
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0146494, 3.0135832
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0767422, 3.0773978
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7566013, 3.7578392
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7256007, 2.7258279

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403154, upper bound: 1.3384744
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3392285, upper bound: 1.3395586
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3309898, 3.3317122
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6166058, 3.6211939
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0577440, 4.0578313
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7645655, 3.7653961
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2022123, 3.2006059
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0141392, 3.0140944
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0763121, 3.0778270
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7588634, 3.7555780
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7251878, 2.7262414

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3405740, upper bound: 1.3373930
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3384036, upper bound: 1.3395647
time: 5.01 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3416526, upper bound: 1.3395577
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3417960, upper bound: 1.3394144
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3396193, upper bound: 1.3398043
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3377045, upper bound: 1.3417199
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3388247, upper bound: 1.3394907
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3377400, upper bound: 1.3405762
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3386650, upper bound: 1.3400743
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3386650, upper bound: 1.3392903
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3403824, upper bound: 1.3376575
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3382154, upper bound: 1.3376624
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3381998, upper bound: 1.3394818
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3389836, upper bound: 1.3394819
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3403154, upper bound: 1.3384744
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3392285, upper bound: 1.3395586
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3405740, upper bound: 1.3373930
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 6, lower bound: -1.3384036, upper bound: 1.3395647

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3043542, 3.3023272
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4575820, 3.4508157
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6682358, 3.6635203
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0167217, 4.0118246
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6907558, 3.6845913
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1895881, 3.1884856
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0947800, 3.0906701
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7169123, 3.7242851
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6043730, 2.5909817

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 946

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3413217, upper bound: 1.3392953
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3413217, upper bound: 1.3384847
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3043056, 3.3023753
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4551473, 3.4532502
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6703901, 3.6613650
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0135002, 4.0150461
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6893950, 3.6859531
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1899981, 3.1880760
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0506096, 3.0506096
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0938568, 3.0915937
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7187366, 3.7224598
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6029158, 2.5924392

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3410972, upper bound: 1.3394135
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3417951, upper bound: 1.3387158
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3047047, 3.2990894
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4659004, 3.4545472
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6775541, 3.6721816
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9904165, 3.9942098
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6983843, 3.7032404
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1823215, 3.1839652
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0450120, 3.0433288
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0848370, 3.0798202
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7079229, 3.7050500
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5924554, 2.5928245

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 877

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3392616, upper bound: 1.3394399
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3384680, upper bound: 1.3394396
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3049498, 3.2988443
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.4714375, 3.4490113
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6815443, 3.6681905
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -3.9880762, 3.9965510
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.6977339, 3.7038918
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1818771, 3.1844087
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0448289, 3.0435128
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0845604, 3.0800977
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7043781, 3.7085953
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.5881400, 2.5971391

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3375605, upper bound: 1.3417195
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3377039, upper bound: 1.3415766
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3072486, 3.3077130
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5193539, 3.5249493
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6993246, 3.7082129
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0472708, 4.0458355
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7632332, 3.7624860
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2012501, 3.2017870
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0252361, 3.0255580
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0888510, 3.0902319
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7491236, 3.7485180
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7049122, 2.7065854

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3388174, upper bound: 1.3375704
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3369089, upper bound: 1.3394836
time: 6.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3072515, 3.3077102
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5193558, 3.5249467
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6993246, 3.7082124
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0472708, 4.0458364
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7632332, 3.7624865
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2012501, 3.2017875
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0252361, 3.0255580
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0888519, 3.0902305
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7491236, 3.7485180
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7049122, 2.7065854

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 5791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3366552, upper bound: 1.3384076
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3366497, upper bound: 1.3405740
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3064346, 3.3078794
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5160160, 3.5236943
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6993432, 3.7082372
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0469942, 4.0462222
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7625294, 3.7621303
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2011108, 3.2006502
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0247946, 3.0243130
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0869741, 3.0886173
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7492113, 3.7481594
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7035556, 2.7061639

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 150

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 508

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3386625, upper bound: 1.3378968
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3364872, upper bound: 1.3400718
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3060036, 3.3083072
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5156555, 3.5240479
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.6995206, 3.7080607
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0473404, 4.0458741
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7624350, 3.7622232
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2010670, 3.2005892
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0249224, 3.0241847
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0865841, 3.0890036
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7494726, 3.7478976
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7034278, 2.7062883

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3386650, upper bound: 1.3381982
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3375730, upper bound: 1.3392907
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3281431, 3.3299518
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5846672, 3.5902386
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7870884, 3.7881727
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0167685, 4.0120497
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7104425, 3.7049036
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1895585, 3.1887412
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0338449, 3.0371013
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0717735, 3.0729852
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7337418, 3.7358575
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6358500, 2.6271172

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5791

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403052, upper bound: 1.3376494
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3403043, upper bound: 1.3373957
time: 8.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3300838, 3.3280106
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5909462, 3.5839586
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7883339, 3.7869267
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0128632, 4.0159550
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7052870, 3.7100582
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1877580, 3.1905408
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0367231, 3.0342231
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0725517, 3.0722079
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7361794, 3.7334185
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6275406, 2.6354263

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3382079, upper bound: 1.3357457
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3362931, upper bound: 1.3376556
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3337173, 3.3331499
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6257396, 3.6246676
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0597172, 4.0592527
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7668591, 3.7663832
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2046700, 3.2056098
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0149274, 3.0154333
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0768099, 3.0768523
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7630110, 3.7629509
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7268839, 2.7263334

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3378476, upper bound: 1.3387279
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3374457, upper bound: 1.3391294
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3332882, 3.3335810
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6253810, 3.6250281
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0600662, 4.0589056
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7667656, 3.7664776
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2047310, 3.2056537
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0150552, 3.0153046
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0764227, 3.0772424
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7632732, 3.7626886
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7267561, 2.7264607

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 5791
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3389760, upper bound: 1.3375690
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3370702, upper bound: 1.3394745
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3304319, 3.3322735
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6194439, 3.6183290
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0583372, 4.0572295
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7653379, 3.7646213
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2006807, 3.2021317
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0146484, 3.0135818
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0767374, 3.0773935
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7565985, 3.7578368
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7256007, 2.7258275

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 508
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 150

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3399559, upper bound: 1.3377128
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3395538, upper bound: 1.3381151
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3304348, 3.3322701
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.6194458, 3.6183262
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7942100, 3.7942100
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0583363, 4.0572305
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7653379, 3.7646217
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.2006798, 3.2021327
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0146484, 3.0135818
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0767384, 3.0773921
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7565985, 3.7578373
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.7256007, 2.7258272

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 508

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3390831, upper bound: 1.3395583
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3392279, upper bound: 1.3394138
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.5452795, -5.7085724, -9.5452795, -5.7085724, -3.3258457, 3.3285093
1: -13.2111492, -8.7825651, -13.2111492, -8.7825651, -3.5743170, 3.5851836
2: -8.1306305, -4.3364205, -8.1306305, -4.3364205, -3.7842131, 3.7835832
3: -9.8012695, -5.1651030, -9.8012695, -5.1651030, -4.0147896, 4.0109725
4: -11.0695591, -7.0785904, -11.0695591, -7.0785904, -3.7078304, 3.7035065
5: -0.2625299, 3.1949592, -0.2625299, 3.1949592, -3.1868253, 3.1834197
6: 4.4642324, 7.5148420, 4.4642324, 7.5148420, -3.0325012, 3.0353355
7: -18.0411415, -14.2939434, -18.0411415, -14.2939434, -3.0699654, 3.0722570
8: 0.0874861, 4.0993404, 0.0874861, 4.0993404, -3.7295914, 3.7287450
9: -8.9012699, -5.7180557, -8.9012699, -5.7180557, -2.6337290, 2.6264734

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 150
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3405670, upper bound: 1.3373876
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.3384007, upper bound: 1.3373916
time: 4.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3413217, upper bound: 1.3392953
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3413217, upper bound: 1.3384847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3410972, upper bound: 1.3394135
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3417951, upper bound: 1.3387158
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3392616, upper bound: 1.3394399
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3384680, upper bound: 1.3394396
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3375605, upper bound: 1.3417195
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3377039, upper bound: 1.3415766
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3388174, upper bound: 1.3375704
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3369089, upper bound: 1.3394836
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3366552, upper bound: 1.3384076
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3366497, upper bound: 1.3405740
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3386625, upper bound: 1.3378968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3364872, upper bound: 1.3400718
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3386650, upper bound: 1.3381982
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3375730, upper bound: 1.3392907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3403052, upper bound: 1.3376494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3403043, upper bound: 1.3373957
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3382079, upper bound: 1.3357457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3362931, upper bound: 1.3376556
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3378476, upper bound: 1.3387279
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3374457, upper bound: 1.3391294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3389760, upper bound: 1.3375690
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3370702, upper bound: 1.3394745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3399559, upper bound: 1.3377128
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3395538, upper bound: 1.3381151
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3390831, upper bound: 1.3395583
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3392279, upper bound: 1.3394138
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3405670, upper bound: 1.3373876
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.58
Output dim: 6, lower bound: -1.3384007, upper bound: 1.3373916
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.58
Output dim: 6, lower bound: -1.3384036, upper bound: 1.3395647
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.0204544067382812
rel_dist={6: [-1.3425452667053426, 1.3425424335470915]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2556.78 seconds
