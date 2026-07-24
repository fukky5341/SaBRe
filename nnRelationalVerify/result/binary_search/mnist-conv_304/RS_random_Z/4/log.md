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
execution time: IAR + LP analysis = 14.62 + 34.86 = 49.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4068807, upper bound: 2.4068790


# Binary Search by BASE starts (time budget: 3550.52 seconds, max iter: 100)

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
Binary search time: 217.65 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3332.86 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9655236, upper bound: 1.9691948
time: 13.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9691954, upper bound: 1.9655228
time: 31.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 44.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 44.62
Output dim: 8, lower bound: -1.9655236, upper bound: 1.9691948
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 44.62
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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9655192, upper bound: 1.9649604
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9612951, upper bound: 1.9691905
time: 8.18 seconds

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

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9624215, upper bound: 1.9655192
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9691910, upper bound: 1.9587506
time: 4.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.70
Output dim: 8, lower bound: -1.9655192, upper bound: 1.9649604
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.70
Output dim: 8, lower bound: -1.9612951, upper bound: 1.9691905
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.70
Output dim: 8, lower bound: -1.9624215, upper bound: 1.9655192
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.70
Output dim: 8, lower bound: -1.9691910, upper bound: 1.9587506

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6497941, 3.6432872
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1430678, 3.1402645
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1525607
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7941866, 3.8039536
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6291556, 2.6312664
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7757707, 3.7779403
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3169565, 3.3182249
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4800873, 3.4793239

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9642562, upper bound: 1.9646751
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9652371, upper bound: 1.9637144
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6491323, 3.6439500
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1440415, 3.1392903
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1520677
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.8088217, 3.7893186
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6301122, 2.6303101
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7786489, 3.7750626
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3244238, 3.3107586
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4793873, 3.4800239

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9545218, upper bound: 1.9691861
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9612909, upper bound: 1.9624171
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6125383, 3.6063199
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1351700, 3.1381550
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1468892, 3.1557083
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7642384, 3.7659469
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6173906, 2.6096039
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7943840, 3.7890677
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3128881, 3.3285036
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4843264, 3.4857368

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9610775, upper bound: 1.9655179
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9624201, upper bound: 1.9641381
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6004753, 3.6183834
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1343775, 3.1389475
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1483941, 3.1542025
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7610788, 3.7691059
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6107583, 2.6162364
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7883596, 3.7950926
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3223057, 3.3190866
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4856739, 3.4843903

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9679401, upper bound: 1.9584660
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9689075, upper bound: 1.9575022
time: 6.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9642562, upper bound: 1.9646751
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9652371, upper bound: 1.9637144
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9545218, upper bound: 1.9691861
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9612909, upper bound: 1.9624171
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9610775, upper bound: 1.9655179
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9624201, upper bound: 1.9641381
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9679401, upper bound: 1.9584660
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.9689075, upper bound: 1.9575022

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6496067, 3.6432185
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1430407, 3.1402016
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1526241
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7949457, 3.8051238
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6287050, 2.6300511
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7752271, 3.7764740
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3168631, 3.3181915
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4791203, 3.4789639

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9574975, upper bound: 1.9646724
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9642520, upper bound: 1.9578963
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6497250, 3.6431003
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1430044, 3.1402373
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1526561
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7953558, 3.8047128
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6279407, 2.6308155
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7743058, 3.7773962
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3169241, 3.3181314
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4797277, 3.4783568

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9602950, upper bound: 1.9587260
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9603010, upper bound: 1.9587170
time: 5.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6198874, 3.6026402
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1421232, 3.1365786
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1546950, 3.1483927
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.8011656, 3.7785034
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6139126, 2.6074781
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7881045, 3.7784939
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3299150, 3.3256683
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4819860, 3.4839697

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9485941, upper bound: 1.9688722
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9542079, upper bound: 1.9632584
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6078234, 3.6147037
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1413298, 3.1373706
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1468873
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7980061, 3.7816625
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6072803, 2.6141105
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7820792, 3.7845187
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3393335, 3.3162513
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4833326, 3.4826231

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9599202, upper bound: 1.9624155
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9612895, upper bound: 1.9610731
time: 4.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6118736, 3.6063828
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1344604, 3.1382241
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1442833, 3.1538620
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7657814, 3.7681026
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6176205, 2.6073339
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7945404, 3.7874918
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3145027, 3.3311577
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4829035, 3.4858770

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9561331, upper bound: 1.9605813
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9561411, upper bound: 1.9605736
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6126022, 3.6056552
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1352386, 3.1374459
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1450424, 3.1531024
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7663937, 3.7674899
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6151204, 2.6098342
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7928085, 3.7892237
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3155422, 3.3301177
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4844675, 3.4843130

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9624157, upper bound: 1.9599201
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9581880, upper bound: 1.9641338
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6002874, 3.6183133
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1343493, 3.1388831
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1484900, 3.1542664
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7618389, 3.7702765
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6103077, 2.6150217
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7878160, 3.7936277
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3222103, 3.3190517
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4847069, 3.4840298

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9679381, upper bound: 1.9579142
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9673911, upper bound: 1.9584636
time: 7.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6004057, 3.6181951
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1343131, 3.1389194
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1484575, 3.1542988
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7622499, 3.7698660
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6095433, 2.6157858
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7868938, 3.7945499
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3222704, 3.3189917
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4853134, 3.4834228

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9629842, upper bound: 1.9571890
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9685961, upper bound: 1.9515761
time: 4.89 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9574975, upper bound: 1.9646724
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9642520, upper bound: 1.9578963
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9602950, upper bound: 1.9587260
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9603010, upper bound: 1.9587170
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9485941, upper bound: 1.9688722
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9542079, upper bound: 1.9632584
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9599202, upper bound: 1.9624155
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9612895, upper bound: 1.9610731
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9561331, upper bound: 1.9605813
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9561411, upper bound: 1.9605736
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9624157, upper bound: 1.9599201
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9581880, upper bound: 1.9641338
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9679381, upper bound: 1.9579142
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9673911, upper bound: 1.9584636
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9629842, upper bound: 1.9571890
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.77
Output dim: 8, lower bound: -1.9685961, upper bound: 1.9515761

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6203623, 3.6019092
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1411209, 3.1374893
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1542978, 3.1489491
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7872877, 3.7943077
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6125064, 2.6072199
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7846823, 3.7799034
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3223524, 3.3330984
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4817200, 3.4829102

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9543905, upper bound: 1.9640772
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9532726, upper bound: 1.9548151
time: 6.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6082983, 3.6139727
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1403294, 3.1382813
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1558027, 3.1474438
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7841282, 3.7974663
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6058741, 2.6138525
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7786570, 3.7859287
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3317699, 3.3236809
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4830666, 3.4815631

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9583275, upper bound: 1.9575855
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9639393, upper bound: 1.9519738
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6496048, 3.6431670
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1420898, 3.1407590
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1519656
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7941628, 3.8053789
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6280646, 2.6305966
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7735353, 3.7778463
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3155947, 3.3188744
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4789762, 3.4787803

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9602930, upper bound: 1.9581719
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9597471, upper bound: 1.9587240
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6497250, 3.6429796
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1430044, 3.1393228
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1526561
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7953558, 3.8035192
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6277218, 2.6308155
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7743058, 3.7766266
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3169241, 3.3168030
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4797277, 3.4776058

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9578762, upper bound: 1.9562868
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9602826, upper bound: 1.9562904
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6181240, 3.5965333
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1375017, 3.1352453
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1501975, 3.1328111
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7863379, 3.7742262
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6099515, 2.6063359
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7866054, 3.7733240
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3176966, 3.3221426
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4815598, 3.4824791

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9473483, upper bound: 1.9685916
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9483092, upper bound: 1.9676245
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6137791, 3.6008787
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1407900, 3.1319585
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1391120, 3.1438956
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7968884, 3.7636762
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6127706, 2.6035171
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7829347, 3.7769947
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3263903, 3.3134484
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4804955, 3.4835439

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9542058, upper bound: 1.9627078
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9536581, upper bound: 1.9632562
time: 5.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6071591, 3.6147680
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1406221, 3.1374407
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1535940, 3.1450405
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7995481, 3.7838163
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6075106, 2.6118402
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7822342, 3.7829409
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3409472, 3.3189049
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4819088, 3.4827633

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9594413, upper bound: 1.9621271
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9594146, upper bound: 1.9605967
time: 5.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6078868, 3.6140394
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1414003, 3.1366620
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1543541, 3.1442809
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.8001604, 3.7832041
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6050100, 2.6143405
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7805023, 3.7846737
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3419867, 3.3178654
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4834738, 3.4811993

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9612875, upper bound: 1.9605235
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9607391, upper bound: 1.9610709
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6117525, 3.6064501
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1335449, 3.1387448
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1446753, 3.1531701
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7645893, 3.7687702
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6177449, 2.6071157
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7937698, 3.7879410
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3131752, 3.3319016
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4821520, 3.4863009

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9530267, upper bound: 1.9599985
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9518905, upper bound: 1.9507394
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6118736, 3.6062622
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1344604, 3.1373091
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1435919, 3.1538620
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7657814, 3.7669106
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6174026, 2.6073339
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7945404, 3.7867212
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3145027, 3.3298302
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4829035, 3.4851265

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9530347, upper bound: 1.9599896
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9518991, upper bound: 1.9507314
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6147685, 3.6071587
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1374407, 3.1406221
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1450405, 3.1535935
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7838163, 3.7995477
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6118402, 2.6075106
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7829409, 3.7822342
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3189058, 3.3409472
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4827633, 3.4819088

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9593071, upper bound: 1.9593307
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9581599, upper bound: 1.9501044
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.6141057, 3.6078210
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1384144, 3.1396480
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1455336, 3.1531005
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7984514, 3.7849121
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6127968, 2.6065540
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7858181, 3.7793570
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3263712, 3.3334804
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4820633, 3.4826088

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9522638, upper bound: 1.9638210
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9578764, upper bound: 1.9582092
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.5933838, 3.6134267
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1289606, 3.1350636
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1451492, 3.1538234
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7653694, 3.7729006
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6064353, 2.6093826
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7813139, 3.7890220
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3201923, 3.3162017
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4803638, 3.4809494

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9673448, upper bound: 1.9579126
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9673494, upper bound: 1.9563389
time: 5.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.5954008, 3.6114097
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1305304, 3.1334944
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1480465, 3.1509261
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.7644634, 3.7738070
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.6046691, 2.6111491
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.7832117, 3.7871251
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.3193607, 3.3170323
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.4816256, 3.4796872

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9667983, upper bound: 1.9584622
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9668029, upper bound: 1.9568813
time: 5.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9543905, upper bound: 1.9640772
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9532726, upper bound: 1.9548151
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9583275, upper bound: 1.9575855
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9639393, upper bound: 1.9519738
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9602930, upper bound: 1.9581719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9597471, upper bound: 1.9587240
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9578762, upper bound: 1.9562868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9602826, upper bound: 1.9562904
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9473483, upper bound: 1.9685916
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9483092, upper bound: 1.9676245
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9542058, upper bound: 1.9627078
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9536581, upper bound: 1.9632562
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9594413, upper bound: 1.9621271
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9594146, upper bound: 1.9605967
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9612875, upper bound: 1.9605235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9607391, upper bound: 1.9610709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9530267, upper bound: 1.9599985
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9518905, upper bound: 1.9507394
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9530347, upper bound: 1.9599896
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9518991, upper bound: 1.9507314
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9593071, upper bound: 1.9593307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9581599, upper bound: 1.9501044
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9522638, upper bound: 1.9638210
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9578764, upper bound: 1.9582092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9673448, upper bound: 1.9579126
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9673494, upper bound: 1.9563389
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9667983, upper bound: 1.9584622
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.74
Output dim: 8, lower bound: -1.9668029, upper bound: 1.9568813
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 8, lower bound: -1.9629842, upper bound: 1.9571890
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 8, lower bound: -1.9685961, upper bound: 1.9515761
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.944035530090332
rel_dist={8: [-1.969206455838302, 1.969206699419745]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6210930, upper bound: 1.6214729
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6214734, upper bound: 1.6210926
time: 6.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.35
Output dim: 8, lower bound: -1.6210930, upper bound: 1.6214729
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.35
Output dim: 8, lower bound: -1.6214734, upper bound: 1.6210926

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2528801, 3.2529869
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8866472, 2.8874679
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9296894, 2.9290695
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4161205, 3.4171829
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2839274, 2.2837319
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4193554, 3.4200530
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9660873, 2.9672704
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7665663, 2.7658975
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1195598, 3.1202302

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6191103, upper bound: 1.6214667
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6210867, upper bound: 1.6194912
time: 6.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2529869, 3.2528801
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8874674, 2.8866472
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9290695, 2.9296894
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4171829, 3.4161205
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2837319, 2.2839274
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4200535, 3.4193559
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9672709, 2.9660869
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7658978, 2.7665665
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1202302, 3.1195593

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6209489, upper bound: 1.6210923
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6214729, upper bound: 1.6205679
time: 7.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.10
Output dim: 8, lower bound: -1.6191103, upper bound: 1.6214667
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.10
Output dim: 8, lower bound: -1.6210867, upper bound: 1.6194912
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.10
Output dim: 8, lower bound: -1.6209489, upper bound: 1.6210923
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.10
Output dim: 8, lower bound: -1.6214729, upper bound: 1.6205679

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2469993, 3.2437663
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8828077, 2.8814702
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9222951, 2.9174953
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4111547, 3.4094357
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2820892, 2.2825530
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4186363, 3.4189277
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9597816, 2.9574237
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7583194, 2.7606173
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1194363, 3.1200719

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6184485, upper bound: 1.6211821
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6188251, upper bound: 1.6208057
time: 10.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2436595, 3.2471061
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8806486, 2.8836284
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9181142, 2.9216747
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4083738, 3.4122171
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2827487, 2.2818935
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4182310, 3.4193325
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9562397, 2.9609656
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7612858, 2.7576506
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1194010, 3.1201081

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6204249, upper bound: 1.6192060
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6208015, upper bound: 1.6188293
time: 10.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2523212, 3.2526298
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8867588, 2.8863831
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9264641, 2.9275174
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4187260, 3.4180126
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2828908, 2.2816575
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4194636, 3.4177771
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9688854, 2.9682956
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7632232, 2.7648206
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1188049, 3.1190281

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6209459, upper bound: 1.6186138
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6184708, upper bound: 1.6210890
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2527370, 3.2522135
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8872042, 2.8859382
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9268980, 2.9270835
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4190760, 3.4176626
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2814622, 2.2830863
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4184737, 3.4187670
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9694786, 2.9677014
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7641511, 2.7638924
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1196995, 3.1181345

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165923, upper bound: 1.6199614
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6208655, upper bound: 1.6156894
time: 5.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6184485, upper bound: 1.6211821
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6188251, upper bound: 1.6208057
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6204249, upper bound: 1.6192060
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6208015, upper bound: 1.6188293
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6209459, upper bound: 1.6186138
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6184708, upper bound: 1.6210890
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6165923, upper bound: 1.6199614
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 8, lower bound: -1.6208655, upper bound: 1.6156894

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2468119, 3.2436466
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8827658, 2.8814073
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9223776, 2.9175601
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4119139, 3.4104295
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2813110, 2.2813377
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4176970, 3.4174619
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9596863, 2.9573627
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7563772, 2.7589970
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1184683, 3.1194506

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6135645, upper bound: 1.6205727
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6178386, upper bound: 1.6162979
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2468796, 3.2435789
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8827457, 2.8814278
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9223595, 2.9175787
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4121485, 3.4101944
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2808743, 2.2817745
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4171696, 3.4179888
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9597206, 2.9573283
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7566996, 2.7586749
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1188154, 3.1191034

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6188237, upper bound: 1.6204812
time: 9.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6185004, upper bound: 1.6208035
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2434721, 3.2469864
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8806076, 2.8835659
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9181986, 2.9217396
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4091320, 3.4132109
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2819700, 2.2806783
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4172916, 3.4178667
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9561443, 2.9609046
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7593436, 2.7560303
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1184320, 3.1194863

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6155423, upper bound: 1.6185970
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6198159, upper bound: 1.6143207
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2435398, 3.2469192
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8805866, 2.8835864
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9181795, 2.9217577
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4093666, 3.4129763
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2815332, 2.2811151
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4167652, 3.4183936
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9561787, 2.9608703
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7596660, 2.7557082
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1187792, 3.1191397

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159181, upper bound: 1.6182186
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6201919, upper bound: 1.6139449
time: 7.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2542048, 3.2541347
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8889608, 2.8891411
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9264631, 2.9277987
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4361458, 3.4437962
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2796097, 2.2789230
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4095993, 3.4095569
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9722490, 2.9759254
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7578773, 2.7556047
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1168013, 3.1166234

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205837, upper bound: 1.6183266
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205593, upper bound: 1.6177312
time: 19.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2538252, 3.2545133
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8895178, 2.8885841
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9267454, 2.9275174
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4445086, 3.4354334
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2801561, 2.2783763
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4112434, 3.4079127
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9765158, 2.9716592
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7540078, 2.7594743
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1164007, 3.1170239

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6135850, upper bound: 1.6204816
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6178593, upper bound: 1.6162088
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2501898, 3.2475281
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8835883, 2.8792844
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9229059, 2.9249067
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4168854, 3.4136376
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2770729, 2.2750404
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4197888, 3.4187622
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9694786, 2.9679041
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7612333, 2.7622993
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1196918, 3.1202860

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6157068, upper bound: 1.6195722
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6163043, upper bound: 1.6195974
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2480516, 3.2496657
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8805499, 2.8823228
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9247217, 2.9230919
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4150505, 3.4154720
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2734156, 2.2786977
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4184690, 3.4200826
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9696827, 2.9677014
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7625580, 2.7609744
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1218510, 3.1181269

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6208624, upper bound: 1.6132044
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6183809, upper bound: 1.6156863
time: 5.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6135645, upper bound: 1.6205727
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6178386, upper bound: 1.6162979
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6188237, upper bound: 1.6204812
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6185004, upper bound: 1.6208035
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6155423, upper bound: 1.6185970
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6198159, upper bound: 1.6143207
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6159181, upper bound: 1.6182186
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6201919, upper bound: 1.6139449
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6205837, upper bound: 1.6183266
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6205593, upper bound: 1.6177312
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6135850, upper bound: 1.6204816
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6178593, upper bound: 1.6162088
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6157068, upper bound: 1.6195722
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6163043, upper bound: 1.6195974
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6208624, upper bound: 1.6132044
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 8, lower bound: -1.6183809, upper bound: 1.6156863

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2442627, 3.2389593
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8791513, 2.8747544
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9183855, 2.9153833
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4097233, 3.4064045
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2769222, 2.2732918
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4190121, 3.4174566
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9596844, 2.9575634
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7534585, 2.7574036
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1184616, 3.1216025

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6116989, upper bound: 1.6205645
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6116974, upper bound: 1.6183314
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2421246, 3.2410965
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8761129, 2.8777933
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9202013, 2.9135685
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4078884, 3.4082394
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2732649, 2.2769492
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4176912, 3.4187765
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9598885, 2.9573607
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7547836, 2.7560787
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1206207, 3.1194434

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6178372, upper bound: 1.6159703
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6175112, upper bound: 1.6162965
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2399750, 3.2378273
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8773565, 2.8769360
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9190168, 2.9158916
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4152899, 3.4128180
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2762451, 2.2761362
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4106679, 3.4125714
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9573460, 2.9544787
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7552776, 2.7569695
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1144729, 3.1154819

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6188207, upper bound: 1.6179986
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6163416, upper bound: 1.6204781
time: 6.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2411280, 3.2366748
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8782530, 2.8760395
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9206724, 2.9142361
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4147720, 3.4133358
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2752352, 2.2771454
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4117532, 3.4114876
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9568710, 2.9549532
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7549939, 2.7572532
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1151938, 3.1147609

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6136129, upper bound: 1.6201954
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6178868, upper bound: 1.6159207
time: 6.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2409220, 3.2422991
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8769922, 2.8769131
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9142065, 2.9195628
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4069414, 3.4091864
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2775817, 2.2726324
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4186068, 3.4178615
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9561424, 2.9611053
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7564254, 2.7544370
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1184254, 3.1216388

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6155409, upper bound: 1.6182672
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6152148, upper bound: 1.6185956
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2387848, 3.2444363
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8739538, 2.8799515
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9160223, 2.9177480
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4051065, 3.4110203
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2739244, 2.2762897
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4172869, 3.4191813
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9563465, 2.9609027
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7577500, 2.7531121
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1205845, 3.1194792

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6198129, upper bound: 1.6118346
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6173303, upper bound: 1.6143174
time: 6.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2409897, 3.2422314
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8769722, 2.8769336
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9141893, 2.9195814
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4071760, 3.4089508
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2771449, 2.2730689
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4180794, 3.4183884
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9561768, 2.9610710
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7567477, 2.7541146
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1187725, 3.1212916

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6152927, upper bound: 1.6179968
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159176, upper bound: 1.6179943
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2388525, 3.2443690
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8739338, 2.8799720
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9160032, 2.9177666
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4053421, 3.4107857
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2734876, 2.2767265
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4167595, 3.4197083
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9563808, 2.9608684
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7580724, 2.7527900
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1209316, 3.1191325

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6201889, upper bound: 1.6114591
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6177057, upper bound: 1.6139420
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2540178, 3.2540150
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8889179, 2.8890777
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9265456, 2.9278622
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4369068, 3.4447918
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2799921, 2.2788687
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4086590, 3.4080896
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9723587, 2.9760709
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7569790, 2.7550297
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1158347, 3.1160045

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6183419, upper bound: 1.6164591
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205749, upper bound: 1.6164616
time: 7.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2540846, 3.2539473
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8888969, 2.8890548
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9265265, 2.9278426
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4371414, 3.4445572
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2795553, 2.2793055
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4081316, 3.4086170
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9723930, 2.9760365
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7573013, 2.7547064
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1161819, 3.1156573

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205579, upper bound: 1.6174019
time: 12.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6202352, upper bound: 1.6177276
time: 10.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2512779, 3.2498269
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8859043, 2.8819323
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9227533, 2.9253402
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4423199, 3.4314098
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2757678, 2.2703309
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4125581, 3.4079070
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9765129, 2.9718590
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7510896, 2.7578809
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1163931, 3.1191745

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6116011, upper bound: 1.6204757
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6135787, upper bound: 1.6184988
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.2491398, 3.2519641
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8828659, 2.8849711
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.9245691, 2.9235253
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.4404860, 3.4332442
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.2721105, 2.2739882
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.4112382, 3.4092269
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.9767151, 2.9716563
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.7524142, 2.7565560
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.1185522, 3.1170158

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6174929, upper bound: 1.6159204
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6174670, upper bound: 1.6153234
time: 5.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6116989, upper bound: 1.6205645
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6116974, upper bound: 1.6183314
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6178372, upper bound: 1.6159703
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6175112, upper bound: 1.6162965
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6188207, upper bound: 1.6179986
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6163416, upper bound: 1.6204781
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6136129, upper bound: 1.6201954
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6178868, upper bound: 1.6159207
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6155409, upper bound: 1.6182672
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6152148, upper bound: 1.6185956
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6198129, upper bound: 1.6118346
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6173303, upper bound: 1.6143174
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6152927, upper bound: 1.6179968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6159176, upper bound: 1.6179943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6201889, upper bound: 1.6114591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6177057, upper bound: 1.6139420
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6183419, upper bound: 1.6164591
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6205749, upper bound: 1.6164616
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6205579, upper bound: 1.6174019
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6202352, upper bound: 1.6177276
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6116011, upper bound: 1.6204757
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6135787, upper bound: 1.6184988
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6174929, upper bound: 1.6159204
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.99
Output dim: 8, lower bound: -1.6174670, upper bound: 1.6153234
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 8, lower bound: -1.6157068, upper bound: 1.6195722
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 8, lower bound: -1.6163043, upper bound: 1.6195974
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 8, lower bound: -1.6208624, upper bound: 1.6132044
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.99
Output dim: 8, lower bound: -1.6183809, upper bound: 1.6156863
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7666475772857666
rel_dist={8: [-1.6237368538932522, 1.6237361751053658]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4947911, upper bound: 1.4950781
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4950785, upper bound: 1.4947903
time: 6.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.87 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.87
Output dim: 8, lower bound: -1.4947911, upper bound: 1.4950781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.87
Output dim: 8, lower bound: -1.4950785, upper bound: 1.4947903

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1201448, 3.1201949
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8023391, 2.8023243
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8518810, 2.8518662
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2972937, 3.2974691
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1662884, 2.1659608
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2970791, 3.2966838
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8507109, 2.8507366
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6824660, 2.6827080
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9988174, 2.9990773

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4630

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4947907, upper bound: 1.4949780
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946914, upper bound: 1.4950775
time: 9.02 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1201954, 3.1201444
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8023238, 2.8023396
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8518658, 2.8518801
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2974691, 3.2972932
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1659608, 2.1662884
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2966833, 3.2970791
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8507366, 2.8507109
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6827078, 2.6824663
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9990778, 2.9988174

Time for backsubstitution: 13.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946059, upper bound: 1.4946199
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4950781, upper bound: 1.4946185
time: 7.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.84
Output dim: 8, lower bound: -1.4947907, upper bound: 1.4949780
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.84
Output dim: 8, lower bound: -1.4946914, upper bound: 1.4950775
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.84
Output dim: 8, lower bound: -1.4946059, upper bound: 1.4946199
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.84
Output dim: 8, lower bound: -1.4950781, upper bound: 1.4946185

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1132402, 3.1141548
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7969513, 2.7976079
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8485389, 2.8497663
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.3003092, 3.3000970
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1614070, 2.1603222
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2905755, 3.2909946
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8482170, 2.8478866
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6809745, 2.6810031
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9944735, 2.9952745

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4926850, upper bound: 1.4931864
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4929989, upper bound: 1.4928729
time: 13.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1141043, 3.1132908
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7976236, 2.7969356
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8497806, 2.8485246
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2999210, 3.3004856
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1606498, 2.1610794
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2913899, 3.2901816
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8478603, 2.8482423
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6807613, 2.6812158
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9950142, 2.9947338

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4908951, upper bound: 1.4944806
time: 9.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4940938, upper bound: 1.4912809
time: 6.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1195307, 3.1197920
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8016148, 2.8019309
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8492613, 2.8495712
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2990122, 3.2990980
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1659226, 2.1651783
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2958493, 3.2955022
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8525572, 2.8529778
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6810799, 2.6815343
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9976535, 2.9980631

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 931

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4908103, upper bound: 1.4940228
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4940086, upper bound: 1.4908242
time: 6.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1198435, 3.1194801
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.8019485, 2.8016305
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8495874, 2.8492746
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2992744, 3.2988358
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1648507, 2.1662500
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2951074, 3.2962441
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8530035, 2.8525324
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6817770, 2.6808381
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9983239, 2.9973927

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4912819, upper bound: 1.4940214
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4944805, upper bound: 1.4908229
time: 6.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4926850, upper bound: 1.4931864
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4929989, upper bound: 1.4928729
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4908951, upper bound: 1.4944806
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4940938, upper bound: 1.4912809
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4908103, upper bound: 1.4940228
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4940086, upper bound: 1.4908242
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4912819, upper bound: 1.4940214
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.06
Output dim: 8, lower bound: -1.4944805, upper bound: 1.4908229

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1131201, 3.1141148
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7960348, 2.7973080
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8483109, 2.8490744
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2991152, 3.2997007
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1613350, 2.1601036
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2898068, 3.2907476
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8468895, 2.8474464
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6807265, 2.6802535
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9937220, 2.9950271

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4888890, upper bound: 1.4925882
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4920877, upper bound: 1.4893896
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1132002, 3.1140347
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7966509, 2.7966924
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8478465, 2.8495393
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2999125, 3.2989035
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1611881, 2.1602502
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2903295, 3.2902250
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8477764, 2.8465590
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6802244, 2.6807554
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9942255, 2.9945235

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913870, upper bound: 1.4928679
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4929927, upper bound: 1.4912612
time: 16.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1110210, 3.1086040
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7932491, 2.7902822
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8457885, 2.8458943
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2972717, 3.2964602
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1553469, 2.1530335
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2923746, 3.2901754
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8478603, 2.8483934
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6778436, 2.6792915
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9950075, 2.9963465

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907236, upper bound: 1.4944801
time: 16.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907249, upper bound: 1.4940104
time: 9.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1094179, 3.1102071
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7909708, 2.7925611
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8471503, 2.8445334
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2958956, 3.2978363
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1526041, 2.1557765
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2913828, 3.2911654
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8480110, 2.8482413
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6788373, 2.6782980
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9966269, 2.9947271

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4915928, upper bound: 1.4912739
time: 11.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4940868, upper bound: 1.4887803
time: 11.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1164474, 3.1151056
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7972403, 2.7952771
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8452673, 2.8469400
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2963638, 3.2950735
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1606193, 2.1571321
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2968349, 3.2954969
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8525553, 2.8531280
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6781616, 2.6796098
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9976459, 2.9996753

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4895755, upper bound: 1.4940205
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4895928, upper bound: 1.4892411
time: 31.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1148443, 3.1167088
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7949610, 2.7975564
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8466291, 2.8455787
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2949877, 3.2964497
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1578760, 2.1598749
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2958450, 3.2964869
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8527079, 2.8529758
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6791553, 2.6786163
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9992652, 2.9980559

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4915076, upper bound: 1.4908170
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4940016, upper bound: 1.4883230
time: 6.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1167593, 3.1147933
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7975740, 2.7949762
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8455935, 2.8466430
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2966261, 3.2948112
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1595473, 2.1582036
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2960911, 3.2962394
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8530016, 2.8526826
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6788588, 2.6789141
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9983172, 2.9990048

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4887809, upper bound: 1.4940143
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4912749, upper bound: 1.4915208
time: 6.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1151562, 3.1163969
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7952948, 2.7972550
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8469553, 2.8452821
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2952499, 3.2961869
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1568046, 2.1609468
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2951012, 3.2972293
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8531542, 2.8525305
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6798525, 2.6779203
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9999366, 2.9973855

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4944783, upper bound: 1.4889726
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4926299, upper bound: 1.4908211
time: 5.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4888890, upper bound: 1.4925882
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4920877, upper bound: 1.4893896
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4913870, upper bound: 1.4928679
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4929927, upper bound: 1.4912612
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4907236, upper bound: 1.4944801
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4907249, upper bound: 1.4940104
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4915928, upper bound: 1.4912739
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4940868, upper bound: 1.4887803
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4895755, upper bound: 1.4940205
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4895928, upper bound: 1.4892411
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4915076, upper bound: 1.4908170
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4940016, upper bound: 1.4883230
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4887809, upper bound: 1.4940143
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4912749, upper bound: 1.4915208
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4944783, upper bound: 1.4889726
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.54
Output dim: 8, lower bound: -1.4926299, upper bound: 1.4908211

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1100359, 3.1094284
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7916613, 2.7906547
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8443189, 2.8464437
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2964659, 3.2956753
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1560321, 2.1520576
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2907906, 3.2907414
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8468876, 2.8475966
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6778083, 2.6783292
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9937153, 2.9966393

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4872788, upper bound: 1.4925820
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4888828, upper bound: 1.4909780
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1084328, 3.1110315
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7893820, 2.7929335
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8456807, 2.8450828
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2950907, 3.2970514
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1532888, 2.1548004
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2898006, 3.2917314
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8470402, 2.8474445
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6788020, 2.6773355
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9953346, 2.9950199

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919161, upper bound: 1.4893895
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919175, upper bound: 1.4889177
time: 8.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1064854, 3.1048150
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7922735, 2.7906961
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8394060, 2.8379636
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2942495, 3.2911544
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1593499, 2.1589065
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2895093, 3.2891006
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8405857, 2.8367109
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6719780, 2.6747336
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9940944, 2.9943647

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4875925, upper bound: 1.4922691
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907908, upper bound: 1.4890698
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1039810, 3.1073198
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7906542, 2.7923150
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8362713, 2.8410983
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2921638, 3.2932405
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1598444, 2.1584120
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2892060, 3.2894044
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8379297, 2.8393674
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6742029, 2.6725087
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9940667, 2.9943919

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891965, upper bound: 1.4906656
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4923952, upper bound: 1.4874660
time: 6.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1103578, 3.1082525
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7925406, 2.7899075
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8431826, 2.8436131
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2988119, 3.2982635
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1553068, 2.1519215
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2915382, 3.2885985
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8496790, 2.8506584
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6762152, 2.6783600
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9935837, 2.9955931

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4886179, upper bound: 1.4926905
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4889317, upper bound: 1.4923746
time: 7.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1106696, 3.1079402
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7928410, 2.7895737
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8434792, 2.8432875
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2990742, 3.2980008
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1542349, 2.1529932
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2907963, 3.2893410
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8501253, 2.8502131
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6769109, 2.6776628
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9942541, 2.9949226

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4882240, upper bound: 1.4940034
time: 8.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907180, upper bound: 1.4915079
time: 11.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1051741, 3.1041012
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7863503, 2.7893500
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8363171, 2.8289499
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2810698, 3.2875319
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1486435, 2.1530240
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2877874, 3.2859964
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8357935, 2.8397489
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6675582, 2.6704473
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9955916, 2.9932361

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4868101, upper bound: 1.4900561
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4915905, upper bound: 1.4900405
time: 6.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.1033115, 3.1059637
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7877588, 2.7879410
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8315668, 2.8337007
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2855911, 3.2830105
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1498518, 2.1518159
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.2862148, 3.2875695
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8395195, 2.8360224
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6709862, 2.6670189
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -2.9951358, 2.9936924

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4940846, upper bound: 1.4869295
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4922361, upper bound: 1.4887781
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0803080, 3.0737958
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7948685, 2.7925658
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8400884, 2.8424063
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2869005, 3.2842574
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1406431, 2.1343026
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.3028536, 3.2989283
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8580465, 2.8626580
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6684675, 2.6711273
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.0002451, 3.0028515

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4895733, upper bound: 1.4921701
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4877249, upper bound: 1.4940187
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -11.0312042, -6.8542714, -11.0312042, -6.8542714, -3.0751381, 3.0789661
1: -9.9745617, -6.7860947, -9.9745617, -6.7860947, -2.7945290, 2.7929053
2: -4.8420544, -1.6859304, -4.8420544, -1.6859304, -2.8407331, 2.8417597
3: -1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912
4: -14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.2855473, 3.2856112
5: -8.5575237, -5.0969090, -8.5575237, -5.0969090, -2.1377897, 2.1371450
6: -12.7730389, -8.5379305, -12.7730389, -8.5379305, -3.3002653, 3.3015103
7: -9.1983776, -5.7004948, -9.1983776, -5.7004948, -2.8620825, 2.8586178
8: 9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.6696768, 2.6699154
9: -7.9733381, -3.6992102, -7.9733381, -3.6992102, -3.0008221, 3.0022745

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 6253
type: RSZ, layer: 1, pos: 4630
type: RSZ, layer: 1, pos: 5845
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.4874871, upper bound: 1.4874516
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.4878009, upper bound: 1.4871356
time: 14.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 37.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4872788, upper bound: 1.4925820
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4888828, upper bound: 1.4909780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4919161, upper bound: 1.4893895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4919175, upper bound: 1.4889177
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4875925, upper bound: 1.4922691
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4907908, upper bound: 1.4890698
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4891965, upper bound: 1.4906656
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4923952, upper bound: 1.4874660
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4886179, upper bound: 1.4926905
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4889317, upper bound: 1.4923746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4882240, upper bound: 1.4940034
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4907180, upper bound: 1.4915079
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4868101, upper bound: 1.4900561
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4915905, upper bound: 1.4900405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4940846, upper bound: 1.4869295
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4922361, upper bound: 1.4887781
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4895733, upper bound: 1.4921701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4877249, upper bound: 1.4940187
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4874871, upper bound: 1.4874516
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 37.63
Output dim: 8, lower bound: -1.4878009, upper bound: 1.4871356
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.63
Output dim: 8, lower bound: -1.4915076, upper bound: 1.4908170
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.63
Output dim: 8, lower bound: -1.4940016, upper bound: 1.4883230
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.63
Output dim: 8, lower bound: -1.4887809, upper bound: 1.4940143
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.63
Output dim: 8, lower bound: -1.4912749, upper bound: 1.4915208
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.63
Output dim: 8, lower bound: -1.4944783, upper bound: 1.4889726
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.63
Output dim: 8, lower bound: -1.4926299, upper bound: 1.4908211
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.6844100952148438
rel_dist={8: [-1.4952889722192388, 1.4952881994988214]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2422.85 seconds
