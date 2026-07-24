## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.487812356
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0312042, -6.8542714, -11.0312042, -6.8542714, -4.1769328, 4.1769328)
1: (-9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1884670, 3.1884670)
2: (-4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1561241)
3: (-1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912)
4: (-14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.9799395, 3.9799395)
5: (-8.5575237, -5.0969090, -8.5575237, -5.0969090, -3.2181473, 3.2181475)
6: (-12.7730389, -8.5379305, -12.7730389, -8.5379305, -4.2351084, 4.2351084)
7: (-9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.4978828, 3.4978828)
8: (9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355)
9: (-7.9733381, -3.6992102, -7.9733381, -3.6992102, -4.0845170, 4.0845165)

## BASE Result
execution time: IAR + LP analysis = 15.06 + 35.02 = 50.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4068807, upper bound: 2.4068790


# Binary Search by BASE starts (time budget: 3549.92 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.9311227798461914
rel_dist={8: [-1.8610607868461848, 1.8610605376350104]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.6844100952148438
rel_dist={8: [-1.4952889722192388, 1.4952881994988214]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.519934892654419
rel_dist={8: [-1.2056374419199969, 1.2056392133504037]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.602172374725342
rel_dist={8: [-1.3553178896207374, 1.355318405775778]}

## Binary Search Result
Binary search time: 222.20 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3327.72 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9655236, upper bound: 1.9691948
time: 17.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9691954, upper bound: 1.9655228
time: 42.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 60.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 60.57
Output dim: 8, lower bound: -1.9655236, upper bound: 1.9691948
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 60.57
Output dim: 8, lower bound: -1.9691954, upper bound: 1.9655228

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6476274, 3.6417823
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1408663, 3.1370883
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1520677
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7767649, 3.7718973
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6324358, 2.6335902
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7856369, 3.7849283
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3135948, 3.3073974
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4817915, 3.4817286

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9655216, upper bound: 1.9686438
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9649739, upper bound: 1.9691934
time: 6.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6417823, 3.6476269
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1370878, 3.1408658
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1520677, 3.1561241
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7718973, 3.7767649
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6335902, 2.6324360
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7849274, 3.7856369
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3073978, 3.3135958
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4817286, 3.4817915

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9691930, upper bound: 1.9649738
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9686436, upper bound: 1.9655214
time: 5.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 8, lower bound: -1.9655216, upper bound: 1.9686438
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 8, lower bound: -1.9649739, upper bound: 1.9691934
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 8, lower bound: -1.9691930, upper bound: 1.9649738
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 8, lower bound: -1.9686436, upper bound: 1.9655214

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6407223, 3.6368952
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1354790, 3.1332698
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1560402, 3.1516237
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7802954, 3.7745209
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6285634, 2.6279507
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7791357, 3.7803249
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3115778, 3.3045478
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4774485, 3.4786468

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9641404, upper bound: 1.9686423
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9655202, upper bound: 1.9672860
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6427402, 3.6348777
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1370478, 3.1317005
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1487265
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7793884, 3.7754278
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6267967, 2.6297174
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7810335, 3.7784271
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3107462, 3.3053789
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4787102, 3.4773850

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9635980, upper bound: 1.9691918
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9649725, upper bound: 1.9678283
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6348782, 3.6427398
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1317005, 3.1370478
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1487265, 3.1561241
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7754278, 3.7793889
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6297174, 2.6267967
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7784271, 3.7810335
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3053789, 3.3107462
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4773846, 3.4787097

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9678283, upper bound: 1.9649729
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9691920, upper bound: 1.9635980
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6368961, 3.6407228
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1332703, 3.1354785
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1516237, 3.1560402
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7745209, 3.7802954
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6279507, 2.6285632
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7803249, 3.7791357
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3045492, 3.3115773
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4786472, 3.4774480

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9672859, upper bound: 1.9655204
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9686422, upper bound: 1.9641404
time: 5.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9641404, upper bound: 1.9686423
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9655202, upper bound: 1.9672860
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9635980, upper bound: 1.9691918
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9649725, upper bound: 1.9678283
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9678283, upper bound: 1.9649729
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9691920, upper bound: 1.9635980
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9672859, upper bound: 1.9655204
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.72
Output dim: 8, lower bound: -1.9686422, upper bound: 1.9641404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6400590, 3.6369591
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1347704, 3.1333399
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1534348, 3.1497779
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7818375, 3.7766762
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6287928, 2.6256795
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7792907, 3.7787471
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3131914, 3.3072023
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4760256, 3.4787889

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9582159, upper bound: 1.9683294
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9638276, upper bound: 1.9627152
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6407866, 3.6362309
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1355486, 3.1325617
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1541948, 3.1490183
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7824497, 3.7760634
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6262922, 2.6281800
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7775588, 3.7804794
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3142319, 3.3061628
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4775896, 3.4772243

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9595926, upper bound: 1.9669732
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9652063, upper bound: 1.9613616
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6420760, 3.6349421
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1363392, 3.1317701
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1468806
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7809315, 3.7775826
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6270261, 2.6274462
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7811885, 3.7768497
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3123608, 3.3080330
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4772873, 3.4775271

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9576735, upper bound: 1.9688779
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632853, upper bound: 1.9632639
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6428037, 3.6342134
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1371174, 3.1309919
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1461205
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7815437, 3.7769699
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6245255, 2.6299465
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7794557, 3.7785821
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3134012, 3.3069935
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4788523, 3.4759626

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9590449, upper bound: 1.9675155
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9646587, upper bound: 1.9619040
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6342139, 3.6428037
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1309919, 3.1371174
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1461210, 3.1561241
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7769699, 3.7815437
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6299467, 2.6245255
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7785821, 3.7794557
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3069935, 3.3134003
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4759626, 3.4788518

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9619039, upper bound: 1.9646588
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9675156, upper bound: 1.9590452
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6349416, 3.6420755
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1317701, 3.1363392
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1468811, 3.1561241
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7775822, 3.7809310
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6274462, 2.6270261
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7768493, 3.7811880
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3080339, 3.3123608
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4775267, 3.4772873

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632637, upper bound: 1.9632848
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688776, upper bound: 1.9576730
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6362309, 3.6407866
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1325617, 3.1355481
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1490183, 3.1541948
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7760639, 3.7824502
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6281800, 2.6262920
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7804790, 3.7775583
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3061628, 3.3142314
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4772243, 3.4775901

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9613616, upper bound: 1.9652065
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9669732, upper bound: 1.9595930
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6369596, 3.6400585
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1333399, 3.1347694
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1497784, 3.1534348
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7766762, 3.7818375
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6256795, 2.6287925
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7787471, 3.7792907
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3072023, 3.3131914
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4787884, 3.4760256

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627152, upper bound: 1.9638271
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9683291, upper bound: 1.9582154
time: 5.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9582159, upper bound: 1.9683294
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9638276, upper bound: 1.9627152
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9595926, upper bound: 1.9669732
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9652063, upper bound: 1.9613616
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9576735, upper bound: 1.9688779
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9632853, upper bound: 1.9632639
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9590449, upper bound: 1.9675155
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9646587, upper bound: 1.9619040
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9619039, upper bound: 1.9646588
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9675156, upper bound: 1.9590452
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9632637, upper bound: 1.9632848
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9688776, upper bound: 1.9576730
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9613616, upper bound: 1.9652065
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9669732, upper bound: 1.9595930
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9627152, upper bound: 1.9638271
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.73
Output dim: 8, lower bound: -1.9683291, upper bound: 1.9582154

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6382980, 3.6308537
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1301498, 3.1320066
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1489367, 3.1341953
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7670107, 3.7723989
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6248312, 2.6245370
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7777915, 3.7735777
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3009729, 3.3036780
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4755983, 3.4772973

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9582114, upper bound: 1.9640923
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9539979, upper bound: 1.9683248
time: 5.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6339531, 3.6351986
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1334372, 3.1287198
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1378522, 3.1452799
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7775612, 3.7618489
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6276498, 2.6217179
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7741208, 3.7772484
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3096685, 3.2949834
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4745340, 3.4783616

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9638232, upper bound: 1.9584794
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9596095, upper bound: 1.9627106
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6390266, 3.6301255
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1309280, 3.1312284
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1496968, 3.1334357
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7676229, 3.7717867
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6223307, 2.6270375
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7760596, 3.7753096
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3020124, 3.3026385
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4771633, 3.4757333

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9595881, upper bound: 1.9627484
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9553641, upper bound: 1.9669685
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6346817, 3.6344709
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1342154, 3.1279411
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1386123, 3.1445203
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7781734, 3.7612362
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6251497, 2.6242185
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7723889, 3.7789803
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3107080, 3.2939439
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4760990, 3.4767976

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9652019, upper bound: 1.9571366
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9609778, upper bound: 1.9613572
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6403160, 3.6288366
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1317186, 3.1304369
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1518340, 3.1312981
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7661047, 3.7733059
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6230645, 2.6263034
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7796893, 3.7716804
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3001432, 3.3045092
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4768610, 3.4760356

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9576691, upper bound: 1.9646453
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9534555, upper bound: 1.9688729
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6359701, 3.6331816
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1350060, 3.1271505
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1407504, 3.1423826
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7766542, 3.7627559
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6258836, 2.6234846
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7760177, 3.7753506
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3088369, 3.2958145
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4757967, 3.4770999

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632808, upper bound: 1.9590328
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9590672, upper bound: 1.9632590
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6410437, 3.6281080
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1324978, 3.1296587
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1525941, 3.1305385
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7667170, 3.7726936
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6205640, 2.6288037
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7779565, 3.7734127
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3011827, 3.3034692
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4784250, 3.4744711

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9590405, upper bound: 1.9632971
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9548157, upper bound: 1.9675110
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6366978, 3.6324534
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1357841, 3.1263719
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1415095, 3.1416225
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7772665, 3.7621431
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6233830, 2.6259849
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7742858, 3.7770829
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3098764, 3.2947745
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4773607, 3.4755359

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9646543, upper bound: 1.9576857
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9604297, upper bound: 1.9618994
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6324539, 3.6366982
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1263723, 3.1357841
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1416230, 3.1415095
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7621431, 3.7772670
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6259851, 2.6233830
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7770829, 3.7742863
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.2947760, 3.3098764
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4755354, 3.4773602

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9618995, upper bound: 1.9604297
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9576857, upper bound: 1.9646542
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6281080, 3.6410437
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1296587, 3.1324973
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1305385, 3.1525941
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7726936, 3.7667165
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6288042, 2.6205640
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7734122, 3.7779565
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3034697, 3.3011818
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4744711, 3.4784250

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9675111, upper bound: 1.9548154
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632975, upper bound: 1.9590407
time: 5.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6331816, 3.6359701
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1271505, 3.1350060
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1423831, 3.1407495
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7627554, 3.7766542
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6234846, 2.6258833
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7753501, 3.7760181
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.2958155, 3.3088365
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4771004, 3.4757962

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632592, upper bound: 1.9590669
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9590328, upper bound: 1.9632809
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6288366, 3.6403155
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1304369, 3.1317191
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1312985, 3.1518340
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7733059, 3.7661042
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6263037, 2.6230645
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7716804, 3.7796888
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3045092, 3.3001418
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4760351, 3.4768605

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688731, upper bound: 1.9534553
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9646452, upper bound: 1.9576691
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6344709, 3.6346812
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1279421, 3.1342149
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1445203, 3.1386123
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7612362, 3.7781734
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6242185, 2.6251495
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7789798, 3.7723889
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.2939444, 3.3107071
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4767981, 3.4760985

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9613572, upper bound: 1.9609778
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9571368, upper bound: 1.9652019
time: 4.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9582114, upper bound: 1.9640923
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9539979, upper bound: 1.9683248
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9638232, upper bound: 1.9584794
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9596095, upper bound: 1.9627106
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9595881, upper bound: 1.9627484
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9553641, upper bound: 1.9669685
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9652019, upper bound: 1.9571366
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9609778, upper bound: 1.9613572
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9576691, upper bound: 1.9646453
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9534555, upper bound: 1.9688729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9632808, upper bound: 1.9590328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9590672, upper bound: 1.9632590
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9590405, upper bound: 1.9632971
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9548157, upper bound: 1.9675110
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9646543, upper bound: 1.9576857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9604297, upper bound: 1.9618994
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9618995, upper bound: 1.9604297
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9576857, upper bound: 1.9646542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9675111, upper bound: 1.9548154
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9632975, upper bound: 1.9590407
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9632592, upper bound: 1.9590669
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9590328, upper bound: 1.9632809
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9688731, upper bound: 1.9534553
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9646452, upper bound: 1.9576691
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9613572, upper bound: 1.9609778
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.35
Output dim: 8, lower bound: -1.9571368, upper bound: 1.9652019
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.9669732, upper bound: 1.9595930
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.9627152, upper bound: 1.9638271
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.9683291, upper bound: 1.9582154
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.944035530090332
rel_dist={8: [-1.969206455838302, 1.969206699419745]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6217542, upper bound: 1.6237300
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237306, upper bound: 1.6217540
time: 8.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.01
Output dim: 8, lower bound: -1.6217542, upper bound: 1.6237300
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.01
Output dim: 8, lower bound: -1.6237306, upper bound: 1.6217540

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2471194, 3.2437792
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8837233, 2.8815646
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9223657, 2.9181857
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4123478, 3.4095659
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2821121, 2.2827718
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4194050, 3.4190001
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9611111, 2.9575686
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7584009, 2.7613673
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1201868, 3.1201510

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6217528, upper bound: 1.6234066
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6212300, upper bound: 1.6237295
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6217537, upper bound: 1.6232059
time: 5.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2437797, 3.2471194
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8815641, 2.8837233
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9181867, 2.9223652
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4095659, 3.4123473
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2827716, 2.2821124
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4190006, 3.4194050
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9575691, 2.9611106
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7613673, 2.7584004
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1201515, 3.1201873

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237293, upper bound: 1.6214299
time: 8.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6234066, upper bound: 1.6217536
time: 11.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 34.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 34.38
Output dim: 8, lower bound: -1.6212300, upper bound: 1.6237295
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 34.38
Output dim: 8, lower bound: -1.6217537, upper bound: 1.6232059
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 34.38
Output dim: 8, lower bound: -1.6237293, upper bound: 1.6214299
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 34.38
Output dim: 8, lower bound: -1.6234066, upper bound: 1.6217536

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2464561, 3.2435317
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8830152, 2.8813009
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9197588, 2.9160137
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4138889, 3.4114575
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2812700, 2.2805007
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4188166, 3.4174228
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9627247, 2.9597764
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7557263, 2.7596211
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1187620, 3.1196208

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6212287, upper bound: 1.6234055
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6209059, upper bound: 1.6237282
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2468719, 3.2431154
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8834596, 2.8808560
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9201937, 2.9155793
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4142389, 3.4111075
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2798409, 2.2819295
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4178267, 3.4184127
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9633179, 2.9591823
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7566543, 2.7586930
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1196566, 3.1187267

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6217523, upper bound: 1.6228846
time: 10.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6214289, upper bound: 1.6232052
time: 12.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2368755, 3.2413678
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8761768, 2.8792324
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9148445, 2.9206791
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4127083, 3.4149714
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2781420, 2.2764730
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4124985, 3.4139881
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9551935, 2.9582610
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7599454, 2.7566953
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1158075, 3.1165648

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6232052, upper bound: 1.6214289
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237288, upper bound: 1.6209062
time: 7.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2380276, 3.2402148
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8770733, 2.8783355
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9165001, 2.9190235
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4121895, 3.4154892
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2771325, 2.2774825
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4135828, 3.4129038
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9547186, 2.9587359
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7596622, 2.7569788
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1165285, 3.1158438

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6228824, upper bound: 1.6217523
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6234061, upper bound: 1.6212289
time: 12.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 33.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6212287, upper bound: 1.6234055
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6209059, upper bound: 1.6237282
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6217523, upper bound: 1.6228846
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6214289, upper bound: 1.6232052
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6232052, upper bound: 1.6214289
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6237288, upper bound: 1.6209062
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6228824, upper bound: 1.6217523
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.72
Output dim: 8, lower bound: -1.6234061, upper bound: 1.6212289

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2395511, 3.2377796
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8776274, 2.8768101
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9164181, 2.9143286
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4170322, 3.4140825
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2766399, 2.2748613
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4123158, 3.4120054
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9603510, 2.9569278
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7543049, 2.7579155
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1144209, 3.1159997

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6179115, upper bound: 1.6233964
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6212191, upper bound: 1.6200890
time: 8.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2407031, 3.2366266
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8785238, 2.8759131
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9180746, 2.9126730
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4165134, 3.4146004
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2756305, 2.2758708
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4134002, 3.4109216
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9598761, 2.9574022
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7540212, 2.7581992
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1151419, 3.1152787

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6175888, upper bound: 1.6237186
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6208963, upper bound: 1.6204110
time: 5.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2399669, 3.2373633
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8780718, 2.8763652
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9168530, 2.9138942
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4173813, 3.4137321
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2752113, 2.2762899
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4113259, 3.4129953
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9609442, 2.9563336
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7552328, 2.7569876
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1153154, 3.1151061

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6184352, upper bound: 1.6228722
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6217427, upper bound: 1.6195646
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2411199, 3.2362108
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8789692, 2.8754683
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9185085, 2.9122386
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4168634, 3.4142504
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2742019, 2.2772994
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4124103, 3.4119115
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9604692, 2.9568081
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7549491, 2.7572713
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1160364, 3.1143851

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6181125, upper bound: 1.6231950
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6214193, upper bound: 1.6198874
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2362103, 3.2411194
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8754683, 2.8789687
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9122391, 2.9185081
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4142504, 3.4168634
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2772994, 2.2742019
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4119115, 3.4124107
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9568090, 2.9604692
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7572713, 2.7549489
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1143847, 3.1160359

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6198880, upper bound: 1.6214193
time: 12.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6231955, upper bound: 1.6181130
time: 9.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2366271, 3.2407036
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8759136, 2.8785238
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9126730, 2.9180741
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4146004, 3.4165134
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2758708, 2.2756305
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4109216, 3.4134007
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9574022, 2.9598756
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7581992, 2.7540209
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1152792, 3.1151423

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6204117, upper bound: 1.6208986
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237192, upper bound: 1.6175892
time: 10.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2373633, 3.2399664
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8763657, 2.8780718
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9138947, 2.9168525
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4137325, 3.4173818
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2762899, 2.2752113
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4129958, 3.4113264
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9563341, 2.9609442
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7569876, 2.7552326
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1151056, 3.1153150

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6195653, upper bound: 1.6217423
time: 10.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6228728, upper bound: 1.6184344
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2377801, 3.2395506
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8768101, 2.8776274
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9143286, 2.9164181
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4140825, 3.4170318
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2748613, 2.2766399
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4120059, 3.4123163
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9569283, 2.9603500
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7579155, 2.7543044
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1160002, 3.1144214

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6200889, upper bound: 1.6212188
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6233965, upper bound: 1.6179114
time: 13.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 35.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6179115, upper bound: 1.6233964
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6212191, upper bound: 1.6200890
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6175888, upper bound: 1.6237186
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6208963, upper bound: 1.6204110
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6184352, upper bound: 1.6228722
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6217427, upper bound: 1.6195646
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6181125, upper bound: 1.6231950
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6214193, upper bound: 1.6198874
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6198880, upper bound: 1.6214193
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6231955, upper bound: 1.6181130
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6204117, upper bound: 1.6208986
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6237192, upper bound: 1.6175892
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6195653, upper bound: 1.6217423
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6228728, upper bound: 1.6184344
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6200889, upper bound: 1.6212188
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.47
Output dim: 8, lower bound: -1.6233965, upper bound: 1.6179114

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2359285, 3.2316742
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8730078, 2.8740683
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9071698, 2.8987460
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4022045, 3.4052839
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2726784, 2.2725105
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4092441, 3.4068360
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9481325, 2.9496775
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7430258, 2.7512078
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1135378, 3.1145086

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6179085, upper bound: 1.6209115
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6154329, upper bound: 1.6233927
time: 6.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2334452, 3.2341571
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8748856, 2.8721900
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9008355, 2.9050798
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4082336, 3.3992553
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2742891, 2.2708998
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4071460, 3.4089336
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9530993, 2.9447088
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7475967, 2.7466369
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1129293, 3.1151166

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6212160, upper bound: 1.6176073
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6187384, upper bound: 1.6200858
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2370806, 3.2305212
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8739042, 2.8731713
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9088254, 2.8970904
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4016867, 3.4058018
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2716694, 2.2735200
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4103284, 3.4057517
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9476576, 2.9501519
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7427421, 2.7514915
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1142588, 3.1137877

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6175858, upper bound: 1.6212376
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6151080, upper bound: 1.6237160
time: 6.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2345982, 3.2330046
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8757820, 2.8712931
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9024920, 2.9034243
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4077158, 3.3997736
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2732801, 2.2719092
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4082303, 3.4078493
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9526262, 2.9451838
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7473130, 2.7469203
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1136513, 3.1143956

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6208933, upper bound: 1.6179323
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6184123, upper bound: 1.6204087
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2363443, 3.2312579
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8734522, 2.8736234
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9076037, 2.8983121
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4025545, 3.4049339
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2712498, 2.2739394
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4082541, 3.4078259
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9487257, 2.9490833
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7439537, 2.7502799
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1144323, 3.1136146

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6184322, upper bound: 1.6203898
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159555, upper bound: 1.6228704
time: 7.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2338610, 3.2337408
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8753300, 2.8717451
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9012704, 2.9046459
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4085836, 3.3989053
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2728605, 2.2723284
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4061570, 3.4099236
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9536943, 2.9441152
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7485247, 2.7457087
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1138239, 3.1142230

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6217397, upper bound: 1.6170852
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6192601, upper bound: 1.6195621
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2374973, 3.2301054
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8743486, 2.8727264
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9092593, 2.8966560
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4020367, 3.4054518
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2702403, 2.2749486
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4093385, 3.4067416
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9482508, 2.9495578
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7436700, 2.7505636
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1151533, 3.1128936

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6181094, upper bound: 1.6207154
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6156297, upper bound: 1.6231924
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2350140, 3.2325883
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8762274, 2.8708487
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9029260, 2.9029899
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4080648, 3.3994231
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2718511, 2.2733378
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4072413, 3.4088392
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9532194, 2.9445896
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7482414, 2.7459922
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1145449, 3.1135020

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6214162, upper bound: 1.6174117
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6189341, upper bound: 1.6198849
time: 8.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2325878, 3.2350140
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8708487, 2.8762269
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9029899, 2.9029255
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.3994226, 3.4080653
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2733383, 2.2718511
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4088387, 3.4072409
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9445906, 2.9532189
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7459922, 2.7482414
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1135015, 3.1145449

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6198850, upper bound: 1.6189345
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6174095, upper bound: 1.6214162
time: 9.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2301054, 3.2374969
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8727264, 2.8743486
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8966565, 2.9092593
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4054518, 3.4020367
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2749491, 2.2702403
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4067416, 3.4093385
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9495592, 2.9482508
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7505636, 2.7436700
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1128941, 3.1151528

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6231925, upper bound: 1.6156297
time: 9.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6207159, upper bound: 1.6181093
time: 7.95 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 31.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6179085, upper bound: 1.6209115
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6154329, upper bound: 1.6233927
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6212160, upper bound: 1.6176073
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6187384, upper bound: 1.6200858
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6175858, upper bound: 1.6212376
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6151080, upper bound: 1.6237160
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6208933, upper bound: 1.6179323
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6184123, upper bound: 1.6204087
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6184322, upper bound: 1.6203898
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6159555, upper bound: 1.6228704
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6217397, upper bound: 1.6170852
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6192601, upper bound: 1.6195621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6181094, upper bound: 1.6207154
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6156297, upper bound: 1.6231924
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6214162, upper bound: 1.6174117
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6189341, upper bound: 1.6198849
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6198850, upper bound: 1.6189345
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6174095, upper bound: 1.6214162
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6231925, upper bound: 1.6156297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 31.89
Output dim: 8, lower bound: -1.6207159, upper bound: 1.6181093
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.89
Output dim: 8, lower bound: -1.6204117, upper bound: 1.6208986
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.89
Output dim: 8, lower bound: -1.6237192, upper bound: 1.6175892
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.89
Output dim: 8, lower bound: -1.6195653, upper bound: 1.6217423
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.89
Output dim: 8, lower bound: -1.6228728, upper bound: 1.6184344
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.89
Output dim: 8, lower bound: -1.6200889, upper bound: 1.6212188
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.89
Output dim: 8, lower bound: -1.6233965, upper bound: 1.6179114
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7666475772857666
rel_dist={8: [-1.6237368538932522, 1.6237361751053658]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936706, upper bound: 1.4952828
time: 11.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952829, upper bound: 1.4936696
time: 6.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.55
Output dim: 8, lower bound: -1.4936706, upper bound: 1.4952828
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.55
Output dim: 8, lower bound: -1.4952829, upper bound: 1.4936696

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1136165, 3.1111116
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7980089, 2.7963901
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8433595, 2.8402252
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2908745, 3.2887888
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1653376, 2.1658323
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2973270, 3.2970238
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8436146, 2.8409591
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6761632, 2.6783881
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9996519, 2.9996252

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936701, upper bound: 1.4951844
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935716, upper bound: 1.4952818
time: 7.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1111121, 3.1136169
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7963896, 2.7980094
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8402257, 2.8433599
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2887888, 3.2908745
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1658320, 2.1653376
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2970238, 3.2973275
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8409595, 2.8436151
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6783881, 2.6761630
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9996252, 2.9996524

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952825, upper bound: 1.4935705
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4951844, upper bound: 1.4936699
time: 14.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 34.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 34.61
Output dim: 8, lower bound: -1.4936701, upper bound: 1.4951844
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 34.61
Output dim: 8, lower bound: -1.4935716, upper bound: 1.4952818
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 34.61
Output dim: 8, lower bound: -1.4952825, upper bound: 1.4935705
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 34.61
Output dim: 8, lower bound: -1.4951844, upper bound: 1.4936699

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1067123, 3.1050715
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7926216, 2.7916751
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8400183, 2.8381252
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2938871, 3.2914128
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1604552, 2.1601930
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2908268, 3.2913361
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8411226, 2.8381095
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6746707, 2.6766825
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9953089, 2.9958224

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932784, upper bound: 1.4951835
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936697, upper bound: 1.4947928
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1075764, 3.1042075
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7932940, 2.7910028
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8412600, 2.8368835
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2934990, 3.2918015
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1596985, 2.1609502
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2916393, 3.2905226
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8407660, 2.8384652
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6744576, 2.6768951
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9958496, 2.9952812

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931799, upper bound: 1.4952812
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935712, upper bound: 1.4948899
time: 6.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1042080, 3.1075768
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7910023, 2.7932940
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8368835, 2.8412600
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2918015, 3.2934990
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1609502, 2.1596985
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2905226, 3.2916398
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8384657, 2.8407655
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6768956, 2.6744576
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9952812, 2.9958496

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4948906, upper bound: 1.4935702
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952821, upper bound: 1.4931789
time: 5.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1050720, 3.1067123
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7916756, 2.7926216
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8381252, 2.8400183
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2914124, 3.2938871
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1601930, 2.1604555
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2913361, 3.2908268
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8381090, 2.8411217
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6766829, 2.6746702
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9958229, 2.9953089

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4947926, upper bound: 1.4936690
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4951840, upper bound: 1.4932780
time: 5.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4932784, upper bound: 1.4951835
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4936697, upper bound: 1.4947928
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4931799, upper bound: 1.4952812
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4935712, upper bound: 1.4948899
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4948906, upper bound: 1.4935702
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4952821, upper bound: 1.4931789
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4947926, upper bound: 1.4936690
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.92
Output dim: 8, lower bound: -1.4951840, upper bound: 1.4932780

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1060481, 3.1047196
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7919130, 2.7913003
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8374128, 2.8358455
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2954292, 3.2932177
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1592560, 2.1579218
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2899909, 3.2897587
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8427362, 2.8401694
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6719961, 2.6747046
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9938860, 2.9950705

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907751, upper bound: 1.4951770
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932713, upper bound: 1.4926803
time: 6.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1063600, 3.1044073
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7922468, 2.7909665
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8377390, 2.8355198
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2956924, 3.2929554
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1581841, 2.1589932
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2892489, 3.2905006
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8431826, 2.8397241
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6726923, 2.6740084
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9945564, 2.9944000

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911667, upper bound: 1.4947862
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936627, upper bound: 1.4922894
time: 5.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1069131, 3.1038551
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7925863, 2.7906275
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8386545, 2.8346038
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2950411, 3.2936063
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1584988, 2.1586790
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2908044, 3.2889452
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8423805, 2.8405256
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6717834, 2.6749172
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9944267, 2.9945292

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906774, upper bound: 1.4952743
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931729, upper bound: 1.4927783
time: 7.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1072249, 3.1035433
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7929192, 2.7902937
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8389807, 2.8342781
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2953033, 3.2933435
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1574273, 2.1597505
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2900624, 3.2896876
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8428259, 2.8400803
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6724796, 2.6742210
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9950972, 2.9938588

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4910687, upper bound: 1.4948829
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935642, upper bound: 1.4923872
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1035428, 3.1072245
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7902937, 2.7929192
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8342781, 2.8389802
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2933435, 3.2953038
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1597505, 2.1574273
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2896876, 3.2900620
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8400803, 2.8428259
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6742210, 2.6724794
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9938593, 2.9950976

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4923876, upper bound: 1.4935632
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4948836, upper bound: 1.4910684
time: 7.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1038556, 3.1069126
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7906275, 2.7925854
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8346043, 2.8386545
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2936058, 3.2950411
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1586790, 2.1584988
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2889457, 3.2908049
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8405256, 2.8423805
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6749172, 2.6717834
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9945297, 2.9944272

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4927790, upper bound: 1.4931719
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952750, upper bound: 1.4906767
time: 7.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1044078, 3.1063600
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7909670, 2.7922468
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8355198, 2.8377385
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2929554, 3.2956924
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1589932, 2.1581843
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2905011, 3.2892489
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8397245, 2.8431821
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6740084, 2.6726923
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9944000, 2.9945569

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4922896, upper bound: 1.4936618
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4947859, upper bound: 1.4911661
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1047196, 3.1060481
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7913008, 2.7919130
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8358459, 2.8374128
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2932177, 3.2954297
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1579218, 2.1592560
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2897582, 3.2899914
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8401699, 2.8427362
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6747046, 2.6719961
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9950705, 2.9938865

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4926810, upper bound: 1.4932707
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4951770, upper bound: 1.4907745
time: 6.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4907751, upper bound: 1.4951770
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4932713, upper bound: 1.4926803
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4911667, upper bound: 1.4947862
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4936627, upper bound: 1.4922894
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4906774, upper bound: 1.4952743
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4931729, upper bound: 1.4927783
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4910687, upper bound: 1.4948829
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4935642, upper bound: 1.4923872
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4923876, upper bound: 1.4935632
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4948836, upper bound: 1.4910684
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4927790, upper bound: 1.4931719
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4952750, upper bound: 1.4906767
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4922896, upper bound: 1.4936618
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4947859, upper bound: 1.4911661
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4926810, upper bound: 1.4932707
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.31
Output dim: 8, lower bound: -1.4951770, upper bound: 1.4907745

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1018047, 3.0986142
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7872934, 2.7880888
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8265805, 2.8202629
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2806025, 3.2829123
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1552944, 2.1551683
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2863946, 3.2845888
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8305178, 2.8316770
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6607170, 2.6668539
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9928513, 2.9935789

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907729, upper bound: 1.4933313
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4889306, upper bound: 1.4951744
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0999422, 3.1004763
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7887020, 2.7866802
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8218303, 2.8250136
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2851248, 3.2783909
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1565027, 2.1539602
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2848220, 3.2861619
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8342447, 2.8279510
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6641455, 2.6634254
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9923944, 2.9940352

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932691, upper bound: 1.4908357
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4914261, upper bound: 1.4926785
time: 5.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1021166, 3.0983019
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7876272, 2.7877550
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8269067, 2.8199372
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2808657, 3.2826495
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1542225, 2.1562400
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2856526, 3.2853312
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8309641, 2.8312316
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6614132, 2.6661580
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9935217, 2.9929085

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911645, upper bound: 1.4929399
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4893220, upper bound: 1.4947837
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1002550, 3.1001644
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7890348, 2.7863464
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8221564, 2.8246875
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2853870, 3.2781281
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1554308, 2.1550317
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2840791, 3.2869043
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8346891, 2.8275051
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6648417, 2.6627295
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9930658, 2.9933648

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4936605, upper bound: 1.4904444
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918174, upper bound: 1.4922872
time: 5.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1026697, 3.0977497
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7879658, 2.7874160
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8278222, 2.8190212
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2802143, 3.2833004
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1545372, 2.1559253
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2872081, 3.2837758
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8301611, 2.8320332
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6605043, 2.6670666
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9933920, 2.9930382

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906751, upper bound: 1.4934296
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4888320, upper bound: 1.4952723
time: 7.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1008072, 3.0996122
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7893744, 2.7860074
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8230720, 2.8237715
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2847357, 3.2787790
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1557455, 2.1547174
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2856345, 3.2853489
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8338881, 2.8283067
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6639328, 2.6636381
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9929361, 2.9934945

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931706, upper bound: 1.4909335
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913275, upper bound: 1.4927762
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1029816, 3.0974379
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7882986, 2.7870827
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8281484, 2.8186955
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2804766, 3.2830381
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1534657, 2.1569970
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2864652, 3.2845182
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8306074, 2.8315873
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6612005, 2.6663706
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9940624, 2.9923677

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4910664, upper bound: 1.4930384
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4892233, upper bound: 1.4948808
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1011190, 3.0992999
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7897081, 2.7856741
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8233981, 2.8234458
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2849979, 3.2785168
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1546736, 2.1557889
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2848926, 3.2860909
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8343344, 2.8278613
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6646290, 2.6629422
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9936066, 2.9928241

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4935620, upper bound: 1.4905420
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917188, upper bound: 1.4923848
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0993004, 3.1011190
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7856741, 2.7897077
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8234458, 2.8233976
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2785168, 3.2849979
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1557889, 2.1546738
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2860913, 3.2848926
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8278608, 2.8343334
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6629419, 2.6646290
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9928236, 2.9936061

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4923853, upper bound: 1.4917180
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4905428, upper bound: 1.4935609
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0974379, 3.1029816
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7870827, 2.7882991
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8186955, 2.8281479
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2830381, 3.2804766
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1569972, 2.1534657
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2845178, 3.2864656
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8315878, 2.8306069
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6663704, 2.6612005
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9923677, 2.9940624

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4948814, upper bound: 1.4892225
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930388, upper bound: 1.4910660
time: 5.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0996122, 3.1008072
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7860079, 2.7893744
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8237720, 2.8230720
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2787790, 3.2847357
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1547174, 2.1557453
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2853484, 3.2856350
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8283072, 2.8338876
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6636381, 2.6639328
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9934950, 2.9929357

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4927768, upper bound: 1.4913271
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4909342, upper bound: 1.4931696
time: 6.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0977497, 3.1026692
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7874165, 2.7879658
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8190217, 2.8278222
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2833004, 3.2802143
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1559258, 2.1545372
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2837758, 3.2872081
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8320341, 2.8301616
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6670666, 2.6605043
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9930382, 2.9933915

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952728, upper bound: 1.4888311
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934302, upper bound: 1.4906744
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1001644, 3.1002545
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7863464, 2.7890348
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8246875, 2.8221555
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2781286, 3.2853870
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1550322, 2.1554308
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2869039, 3.2840796
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8275061, 2.8346896
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6627293, 2.6648417
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9933653, 2.9930654

Time for backsubstitution: 14.42 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.6844100952148438
rel_dist={8: [-1.4952889722192388, 1.4952881994988214]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2410.53 seconds
