## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.872541919
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.3938293, 2.3938293)
1: (-17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.6634283, 3.6634278)
2: (-3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.7616858, 2.7616858)
3: (-10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9296875, 2.9296875)
4: (-12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.5041504, 3.5041494)
5: (-4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.3018072, 2.3018072)
6: (-3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.5280423, 2.5280423)
7: (-9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.9013386, 3.9013381)
8: (-2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2600141, 2.2600141)
9: (-4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.7319930, 2.7319930)

## BASE Result
execution time: IAR + LP analysis = 15.18 + 33.42 = 48.59 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.027801513671875
rel_dist={0: [-1.1944731507389008, 1.1944727062795257]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8442518711090088
rel_dist={0: [-0.8742898229747036, 0.874290832632278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.7218856811523438
rel_dist={0: [-0.6370174797469685, 0.637020759518645]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7830688953399658
rel_dist={0: [-0.7572591028880957, 0.7572587984536243]}

## Binary Search Result
Binary search time: 215.30 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3336.11 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2894388, upper bound: 1.2945710
time: 8.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945714, upper bound: 1.2894370
time: 5.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.14
Output dim: 0, lower bound: -1.2894388, upper bound: 1.2945710
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.14
Output dim: 0, lower bound: -1.2945714, upper bound: 1.2894370

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0826464, 2.0844946
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1720753, 3.1835680
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6330547, 2.6275709
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9119596, 2.9164019
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0845323, 3.0865145
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2334080, 2.2368844
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4449801, 2.4490221
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5002255, 3.4998331
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2153468, 2.2218444
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4870381, 2.4926181

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 552

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2894304, upper bound: 1.2943768
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2892417, upper bound: 1.2945647
time: 6.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0844946, 2.0826461
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1835680, 3.1720748
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6275706, 2.6330547
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9164019, 2.9119596
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0865149, 3.0845318
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2368841, 2.2334080
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4490218, 2.4449804
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4998326, 3.5002255
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2218447, 2.2153466
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4926181, 2.4870379

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2943389, upper bound: 1.2894366
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945712, upper bound: 1.2892209
time: 5.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 0, lower bound: -1.2894304, upper bound: 1.2943768
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 0, lower bound: -1.2892417, upper bound: 1.2945647
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 0, lower bound: -1.2943389, upper bound: 1.2894366
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 0, lower bound: -1.2945712, upper bound: 1.2892209

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0844312, 2.0843074
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1716328, 3.1877589
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6329622, 2.6284795
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9119425, 2.9165697
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0889463, 3.0860457
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2331657, 2.2391682
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4444661, 2.4538715
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5107985, 3.4987106
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2152181, 2.2230551
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4910994, 2.4921923

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2882115, upper bound: 1.2935205
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2885571, upper bound: 1.2931713
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0824590, 2.0844946
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1720753, 3.1831250
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6330547, 2.6274781
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9119596, 2.9163842
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0840635, 3.0865145
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2334080, 2.2366419
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4449801, 2.4485080
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4991026, 3.4998331
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2153468, 2.2217159
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4866123, 2.4926181

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2871970, upper bound: 1.2919415
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2866151, upper bound: 1.2925219
time: 8.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0942426, 2.0966699
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1685953, 3.1509504
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6197119, 2.6274881
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8925757, 2.8783212
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0391450, 3.0509915
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2165680, 2.2047217
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4384995, 2.4301248
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4922719, 3.4948688
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2183890, 2.2104702
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4483800, 2.4563191

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2933050, upper bound: 1.2885629
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2933050, upper bound: 1.2881668
time: 7.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0985184, 2.0923941
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1624441, 3.1571021
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6220045, 2.6251960
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8827643, 2.8881330
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0529742, 3.0371616
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2081981, 2.2130919
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4341669, 2.4344578
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4944749, 3.4926653
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2169681, 2.2118912
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4618993, 2.4427998

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4603

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945534, upper bound: 1.2794991
time: 13.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2848634, upper bound: 1.2892023
time: 5.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 33.48 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2882115, upper bound: 1.2935205
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2885571, upper bound: 1.2931713
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2871970, upper bound: 1.2919415
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2866151, upper bound: 1.2925219
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2933050, upper bound: 1.2885629
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2933050, upper bound: 1.2881668
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2945534, upper bound: 1.2794991
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.48
Output dim: 0, lower bound: -1.2848634, upper bound: 1.2892023

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0844254, 2.0860627
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1716242, 3.1913176
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6279063, 2.6267688
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9109983, 2.9137731
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0882959, 3.0858245
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2319112, 2.2354643
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4435492, 2.4511673
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5142984, 3.4987049
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2150660, 2.2226079
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4884262, 2.4912875

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2882018, upper bound: 1.2843116
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2790413, upper bound: 1.2935107
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0861859, 2.0843022
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1751909, 3.1877513
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6312513, 2.6234241
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9091463, 2.9156256
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0887270, 3.0853944
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2294621, 2.2379134
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4417620, 2.4529541
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5107937, 3.5022101
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2147708, 2.2229028
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4901948, 2.4895189

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 453

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2885551, upper bound: 1.2931699
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2885556, upper bound: 1.2931669
time: 6.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0824490, 2.0845027
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1717682, 3.1836843
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6345992, 2.6266294
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9114075, 2.9173908
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0842376, 3.0864186
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2328267, 2.2376993
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4446359, 2.4491327
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4990683, 3.4998941
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2155514, 2.2215974
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4872117, 2.4922886

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2828319, upper bound: 1.2847912
time: 9.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2800146, upper bound: 1.2876019
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0824590, 2.0844848
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1720753, 3.1828184
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6322055, 2.6274781
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9119596, 2.9158320
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0839686, 3.0865145
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2334080, 2.2360604
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4449801, 2.4481633
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4991026, 3.4997983
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2152281, 2.2217159
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4862828, 2.4926181

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2863986, upper bound: 1.2925222
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2866149, upper bound: 1.2922911
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0942373, 2.0984247
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1685886, 3.1545095
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6146564, 2.6257775
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8916302, 2.8755231
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0384936, 3.0507708
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2153125, 2.2010176
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4375815, 2.4274194
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4957724, 3.4948640
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2182374, 2.2100234
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4481707, 2.4578779

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2894563, upper bound: 1.2853080
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2900524, upper bound: 1.2847121
time: 6.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0959978, 2.0966649
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1650372, 3.1509428
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6180010, 2.6224327
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8897762, 2.8773756
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0389228, 3.0503402
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2128639, 2.2034667
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4357944, 2.4292064
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4922676, 3.4939680
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2179422, 2.2103183
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4499397, 2.4561100

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2933039, upper bound: 1.2878983
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2930395, upper bound: 1.2881638
time: 8.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0853658, 2.0738349
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1582122, 3.1565099
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6154656, 2.6154692
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8574595, 2.8730612
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0498986, 3.0319443
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.1909728, 2.2024920
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4292655, 2.4309871
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4888611, 3.4847436
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.1970444, 2.1977732
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4502892, 2.4257786

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945503, upper bound: 1.2769868
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920281, upper bound: 1.2794956
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0799594, 2.0792410
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1618514, 3.1528702
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6122775, 2.6186576
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8676915, 2.8628297
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0477567, 3.0340860
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.1975985, 2.1958668
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4306960, 2.4295568
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4865541, 3.4870505
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2028503, 2.1919675
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4448781, 2.4311898

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2805610, upper bound: 1.2891961
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2848582, upper bound: 1.2848929
time: 5.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2882018, upper bound: 1.2843116
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2790413, upper bound: 1.2935107
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2885551, upper bound: 1.2931699
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2885556, upper bound: 1.2931669
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2828319, upper bound: 1.2847912
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2800146, upper bound: 1.2876019
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2863986, upper bound: 1.2925222
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2866149, upper bound: 1.2922911
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2894563, upper bound: 1.2853080
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2900524, upper bound: 1.2847121
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2933039, upper bound: 1.2878983
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2930395, upper bound: 1.2881638
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2945503, upper bound: 1.2769868
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2920281, upper bound: 1.2794956
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2805610, upper bound: 1.2891961
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 0, lower bound: -1.2848582, upper bound: 1.2848929

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0829983, 2.0819631
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1707134, 3.1886973
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6218433, 2.6246636
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9102879, 2.9135265
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0815482, 3.0663419
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2280526, 2.2341206
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4270029, 2.4454246
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5111942, 3.4897833
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2149730, 2.2223408
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4807863, 2.4692647

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2882007, upper bound: 1.2840619
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2879352, upper bound: 1.2843097
time: 5.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0803261, 2.0846353
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1690044, 3.1904058
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6258016, 2.6207056
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9107513, 2.9130635
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0688138, 3.0790768
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2305675, 2.2316060
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4378061, 2.4346216
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5053768, 3.4956007
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2147989, 2.2225146
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4664035, 2.4836476

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2751984, upper bound: 1.2902559
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2756103, upper bound: 1.2896602
time: 6.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0862894, 2.0841119
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1772375, 3.1840119
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6253333, 2.6266131
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9130068, 2.9085236
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0783167, 3.0910368
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2308378, 2.2353613
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4427180, 2.4511874
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5119810, 3.5000057
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2155619, 2.2214427
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4919491, 2.4862752

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2845175, upper bound: 1.2899162
time: 11.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852998, upper bound: 1.2893207
time: 7.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0859957, 2.0843022
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1714516, 3.1877513
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6312513, 2.6175060
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9020433, 2.9156256
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0887270, 3.0749846
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2269096, 2.2379134
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4399953, 2.4529541
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5085888, 3.5022101
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2133107, 2.2229028
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4869509, 2.4895189

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2885545, upper bound: 1.2929028
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2882888, upper bound: 1.2931654
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0825720, 2.0841870
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1716900, 3.1837182
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6346178, 2.6265883
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9111757, 2.9174814
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0836587, 3.0866280
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2325425, 2.2378149
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4444499, 2.4492080
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4991045, 3.4998021
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2155852, 2.2214894
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4872727, 2.4921284

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2828308, upper bound: 1.2845345
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2825679, upper bound: 1.2847882
time: 8.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0821342, 2.0845027
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1717682, 3.1836071
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6345587, 2.6266294
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9114075, 2.9171591
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0842376, 3.0858402
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2328267, 2.2374151
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4446359, 2.4489467
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4989767, 3.4998941
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2154436, 2.2215974
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4870524, 2.4922886

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2797991, upper bound: 1.2876016
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2800144, upper bound: 1.2873809
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0922074, 2.0985081
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1571035, 3.1616945
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6243472, 2.6219120
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8881335, 2.8821931
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0365973, 3.0529742
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2130919, 2.2073743
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4344578, 2.4333072
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4915428, 3.4944415
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2117729, 2.2168398
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4420447, 2.4618993

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4603

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2863780, upper bound: 1.2828122
time: 15.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2766790, upper bound: 1.2925035
time: 6.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0964832, 2.0942323
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1509504, 3.1678462
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6266398, 2.6196198
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8783221, 2.8920050
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0504274, 3.0391443
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2047219, 2.2157445
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4301252, 2.4376402
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4937458, 3.4922385
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2103515, 2.2182608
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4555640, 2.4483800

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2853436, upper bound: 1.2912588
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2857414, upper bound: 1.2912588
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0940447, 2.0990143
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1683083, 3.1553736
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6145215, 2.6261897
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8916321, 2.8755236
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0422463, 3.0495486
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2151294, 2.2015789
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4374104, 2.4279439
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5015364, 3.4929895
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2177458, 2.2115350
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4510117, 2.4569533

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2874157, upper bound: 1.2826811
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2868341, upper bound: 1.2832641
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0942373, 2.0982316
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1685886, 3.1542292
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6146564, 2.6256423
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8916302, 2.8755231
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0372710, 3.0507708
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2153125, 2.2008350
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4375815, 2.4272485
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4938993, 3.4948640
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2182374, 2.2095320
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4472461, 2.4578779

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2900427, upper bound: 1.2755434
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2808519, upper bound: 1.2847041
time: 7.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0960121, 2.0961976
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1653080, 3.1424637
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6180091, 2.6220505
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8900366, 2.8694077
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0359664, 3.0504303
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2129579, 2.2006290
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4351840, 2.4292264
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4844265, 3.4942369
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2179003, 2.2103233
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4447579, 2.4562714

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 552

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2932979, upper bound: 1.2877047
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2877062, upper bound: 1.2878945
time: 8.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0955305, 2.0966649
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1565580, 3.1509428
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6176190, 2.6224327
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8818092, 2.8773756
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0389228, 3.0473838
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2100263, 2.2034667
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4357944, 2.4285960
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4922676, 3.4861264
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2179422, 2.2102766
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4499397, 2.4509284

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2885999, upper bound: 1.2809419
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2858340, upper bound: 1.2837226
time: 7.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0849261, 2.0737836
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1557708, 3.1562471
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6146703, 2.6081030
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8492532, 2.8721747
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0496359, 3.0295293
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.1851802, 2.2018642
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4275622, 2.4307995
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4861345, 3.4844499
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.1968923, 2.1964676
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4499650, 2.4227903

Time for backsubstitution: 14.35 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.088984727859497
rel_dist={0: [-1.2945795095372876, 1.2945797204952658]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9810025, upper bound: 0.9840139
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840145, upper bound: 0.9810025
time: 7.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.19
Output dim: 0, lower bound: -0.9810025, upper bound: 0.9840139
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.19
Output dim: 0, lower bound: -0.9840145, upper bound: 0.9810025

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8990965, 1.9001529
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8815575, 2.8881245
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4143248, 2.4111910
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6861963, 2.6887345
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8368430, 2.8379755
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0956068, 2.0975926
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3130336, 2.3153429
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2599635, 3.2597389
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0735812, 2.0772943
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3383346, 2.3415232

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 552

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9808259, upper bound: 0.9840141
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9810050, upper bound: 0.9838366
time: 11.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9001532, 1.8990965
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8881245, 2.8815570
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4111910, 2.4143248
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6887350, 2.6861963
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8379760, 2.8368425
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0975933, 2.0956061
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3153434, 2.3130333
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2597394, 3.2599635
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0772943, 2.0735812
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3415232, 2.3383346

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9787738, upper bound: 0.9794143
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9787737, upper bound: 0.9787713
time: 7.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 31.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 0, lower bound: -0.9808259, upper bound: 0.9840141
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 0, lower bound: -0.9810050, upper bound: 0.9838366
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 0, lower bound: -0.9787738, upper bound: 0.9794143
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 0, lower bound: -0.9787737, upper bound: 0.9787713

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9088445, 1.9123442
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8639479, 2.8670001
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4064660, 2.4046421
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6581645, 2.6550961
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7894731, 2.7985082
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0717030, 2.0689063
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3006535, 2.3004873
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2524037, 3.2534380
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0695167, 2.0724177
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2940965, 2.3050103

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9785944, upper bound: 0.9824261
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9792376, upper bound: 0.9817831
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9112878, 1.9099009
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8604326, 2.8705153
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4077759, 2.4033322
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6525569, 2.6607027
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7973752, 2.7906051
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0669203, 2.0736892
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2981777, 2.3029633
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2536626, 3.2521791
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0687051, 2.0732298
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3018217, 2.2972851

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9799370, upper bound: 0.9796991
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9768620, upper bound: 0.9827730
time: 7.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8999610, 1.8993521
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8878441, 2.8819304
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4110556, 2.4145026
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6887369, 2.6861963
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8395963, 2.8356204
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0974102, 2.0958488
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3151727, 2.3132613
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2622290, 3.2580891
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0768032, 2.0742347
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3427510, 2.3374107

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809819, upper bound: 0.9794033
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809855, upper bound: 0.9779763
time: 8.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9001532, 1.8989043
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8881245, 2.8812761
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4111910, 2.4141893
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6887350, 2.6861963
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8367534, 2.8368425
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0975933, 2.0954237
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3153434, 2.3128638
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2578650, 3.2599635
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0772943, 2.0730903
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3405991, 2.3383346

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9780721, upper bound: 0.9774274
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9815723, upper bound: 0.9774271
time: 5.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9785944, upper bound: 0.9824261
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9792376, upper bound: 0.9817831
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9799370, upper bound: 0.9796991
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9768620, upper bound: 0.9827730
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9809819, upper bound: 0.9794033
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9809855, upper bound: 0.9779763
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9780721, upper bound: 0.9774274
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 0, lower bound: -0.9815723, upper bound: 0.9774271

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9086518, 1.9125984
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8636684, 2.8673744
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4063311, 2.4048202
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6581659, 2.6550961
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7910929, 2.7972860
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0715203, 2.0691490
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3004832, 2.3007135
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2548928, 3.2515635
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0690255, 2.0730708
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2953234, 2.3040857

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 453

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9785908, upper bound: 0.9824249
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9785931, upper bound: 0.9824219
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9088445, 1.9121513
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8639479, 2.8667202
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4064660, 2.4045074
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6581640, 2.6550961
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7882509, 2.7985082
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0717030, 2.0687237
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3006535, 2.3003161
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2505288, 3.2534380
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0695167, 2.0719264
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2931719, 2.3050103

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9781729, upper bound: 0.9776447
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9750972, upper bound: 0.9807184
time: 6.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9112222, 1.9095855
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8603554, 2.8705015
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4077673, 2.4032900
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6523242, 2.6606541
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7967958, 2.7904756
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0666361, 2.0736334
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2979908, 2.3029258
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2536430, 3.2520862
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0686784, 2.0731225
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3017869, 2.2971241

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9791772, upper bound: 0.9791149
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9761057, upper bound: 0.9791141
time: 9.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9109724, 1.9098353
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8604183, 2.8704381
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4077334, 2.4033237
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6525092, 2.6604700
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7972460, 2.7900252
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0668640, 2.0734053
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2981405, 2.3027768
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2535696, 3.2521596
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0685973, 2.0732036
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3016610, 2.2972500

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9740518, upper bound: 0.9767990
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9739606, upper bound: 0.9827681
time: 8.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8967223, 1.8979795
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8822908, 2.8795800
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4099832, 2.4140499
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6877794, 2.6857915
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8300753, 2.8131094
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0940199, 2.0944145
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3106904, 2.3113661
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2501988, 3.2296505
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0674324, 2.0702765
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3366852, 2.3230572

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 552

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809755, upper bound: 0.9792127
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9807929, upper bound: 0.9793974
time: 7.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8985882, 1.8961134
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8854942, 2.8763771
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4106016, 2.4134300
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6883268, 2.6852388
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8170853, 2.8260989
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0959759, 2.0924582
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3132787, 2.3087778
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2337899, 3.2460599
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0728455, 2.0648634
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3283973, 2.3313448

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765678, upper bound: 0.9735954
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765678, upper bound: 0.9735954
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8962097, 1.8956196
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8954754, 2.8904171
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4116459, 2.4136777
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6919222, 2.6901598
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8364840, 2.8366179
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0993419, 2.0975950
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3099165, 2.3083405
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2567167, 3.2585869
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0717444, 2.0687962
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3360796, 2.3345685

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 552

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9770161, upper bound: 0.9732866
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9739308, upper bound: 0.9763738
time: 14.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8968687, 1.8949609
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8972654, 2.8886266
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4106789, 2.4146459
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6926985, 2.6893830
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8365288, 2.8365741
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0997682, 2.0971730
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3108215, 2.3074362
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2564878, 3.2588158
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0730000, 2.0675402
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3368330, 2.3338151

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9806085, upper bound: 0.9759197
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9800654, upper bound: 0.9764631
time: 9.10 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9785908, upper bound: 0.9824249
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9785931, upper bound: 0.9824219
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9781729, upper bound: 0.9776447
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9750972, upper bound: 0.9807184
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9791772, upper bound: 0.9791149
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9761057, upper bound: 0.9791141
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9740518, upper bound: 0.9767990
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9739606, upper bound: 0.9827681
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9809755, upper bound: 0.9792127
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9807929, upper bound: 0.9793974
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9765678, upper bound: 0.9735954
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9765678, upper bound: 0.9735954
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9770161, upper bound: 0.9732866
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9739308, upper bound: 0.9763738
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9806085, upper bound: 0.9759197
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.45
Output dim: 0, lower bound: -0.9800654, upper bound: 0.9764631

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9086285, 1.9124074
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8632340, 2.8636341
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4004130, 2.4041059
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6573286, 2.6479945
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7806849, 2.7960498
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0712118, 2.0665956
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3002710, 2.2989471
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2546268, 3.2493591
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0688524, 2.0716114
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2949352, 2.3008413

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9776228, upper bound: 0.9809164
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9770855, upper bound: 0.9809166
time: 17.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9084606, 1.9125752
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8599286, 2.8669400
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4056172, 2.3989019
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6510639, 2.6542592
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7898574, 2.7868772
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0689673, 2.0688405
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2987156, 2.3005025
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2526889, 3.2512975
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0675659, 2.0728974
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2920794, 2.3036973

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9785912, upper bound: 0.9811067
time: 10.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9772783, upper bound: 0.9824199
time: 7.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9087789, 1.9118359
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8638706, 2.8667059
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4064574, 2.4044654
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6579313, 2.6550474
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7876706, 2.7983785
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0714188, 2.0686674
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3004675, 2.3002787
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2505093, 3.2533455
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0694909, 2.0718193
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2931376, 2.3048494

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9767323, upper bound: 0.9768440
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9781591, upper bound: 0.9768402
time: 5.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9085290, 1.9120858
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8639336, 2.8666425
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4064240, 2.4044991
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6581154, 2.6548634
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7881207, 2.7979283
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0716467, 2.0684395
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3006163, 2.3001297
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2504368, 3.2534184
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0694098, 2.0719004
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2930117, 2.3049753

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9745135, upper bound: 0.9802233
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9745135, upper bound: 0.9799530
time: 6.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9112175, 1.9105864
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8603468, 2.8684645
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4027123, 2.4001460
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6505852, 2.6578560
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7961454, 2.7900712
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0643311, 2.0699291
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2963071, 2.3002207
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2531257, 3.2520814
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0683990, 2.0726743
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3015776, 2.2979257

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9732046, upper bound: 0.9762132
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9732046, upper bound: 0.9763049
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9122231, 1.9095805
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8623857, 2.8704934
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4046235, 2.3982346
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6495266, 2.6589146
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7963905, 2.7898254
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0629315, 2.0713282
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2952867, 2.3012419
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2536378, 3.2540846
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0682306, 2.0728428
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3025880, 2.2969151

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9794419, upper bound: 0.9762138
time: 9.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9732046, upper bound: 0.9763046
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9083998, 1.9057357
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8586473, 2.8678179
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4016705, 2.3994567
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6518035, 2.6600289
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7841434, 2.7705426
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0630054, 2.0709841
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2815948, 2.2924042
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2479720, 3.2432370
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0684295, 2.0729361
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2878571, 2.2752278

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4603

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9740489, upper bound: 0.9714082
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9686596, upper bound: 0.9767966
time: 9.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9068730, 1.9072633
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8577986, 2.8687940
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4039321, 2.3972602
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6520686, 2.6597638
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7777634, 2.7778196
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0644436, 2.0695472
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2877679, 2.2862310
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2446475, 3.2465625
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0683303, 2.0730355
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2796388, 2.2834485

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9732020, upper bound: 0.9821825
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9734704, upper bound: 0.9821855
time: 8.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8976617, 1.8977923
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8818493, 2.8817859
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4098911, 2.4145303
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6877613, 2.6858797
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8323970, 2.8126411
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0937762, 2.0956149
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3101754, 2.3139164
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2557602, 3.2285290
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0673037, 2.0709133
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3388243, 2.3226321

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809702, upper bound: 0.9730839
time: 9.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9748598, upper bound: 0.9792064
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8965349, 1.8979795
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8822908, 2.8791375
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4099832, 2.4139578
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6877794, 2.6857738
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8296065, 2.8131094
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0940199, 2.0941713
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3106904, 2.3108516
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2490768, 3.2296505
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0674324, 2.0701480
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3362598, 2.3230572

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5778

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9763748, upper bound: 0.9749953
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9763748, upper bound: 0.9749953
time: 6.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8985882, 1.8961129
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8854933, 2.8763800
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4106035, 2.4134295
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6883259, 2.6852393
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8170853, 2.8260982
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0959759, 2.0924587
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3132787, 2.3087778
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2337899, 3.2460618
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0728455, 2.0648630
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3283973, 2.3313448

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9764218, upper bound: 0.9735940
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765677, upper bound: 0.9734481
time: 5.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9776228, upper bound: 0.9809164
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9770855, upper bound: 0.9809166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9785912, upper bound: 0.9811067
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9772783, upper bound: 0.9824199
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9767323, upper bound: 0.9768440
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9781591, upper bound: 0.9768402
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9745135, upper bound: 0.9802233
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9745135, upper bound: 0.9799530
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9732046, upper bound: 0.9762132
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9732046, upper bound: 0.9763049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9794419, upper bound: 0.9762138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9732046, upper bound: 0.9763046
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9740489, upper bound: 0.9714082
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9686596, upper bound: 0.9767966
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9732020, upper bound: 0.9821825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9734704, upper bound: 0.9821855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9809702, upper bound: 0.9730839
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9748598, upper bound: 0.9792064
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9763748, upper bound: 0.9749953
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9763748, upper bound: 0.9749953
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9764218, upper bound: 0.9735940
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.62
Output dim: 0, lower bound: -0.9765677, upper bound: 0.9734481
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.62
Output dim: 0, lower bound: -0.9765678, upper bound: 0.9735954
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.62
Output dim: 0, lower bound: -0.9770161, upper bound: 0.9732866
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.62
Output dim: 0, lower bound: -0.9739308, upper bound: 0.9763738
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.62
Output dim: 0, lower bound: -0.9806085, upper bound: 0.9759197
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.62
Output dim: 0, lower bound: -0.9800654, upper bound: 0.9764631
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.9054350852966309
rel_dist={0: [-0.9840186515659699, 0.9840187223222951]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726915, upper bound: 0.8733362
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733366, upper bound: 0.8726909
time: 5.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.00
Output dim: 0, lower bound: -0.8726915, upper bound: 0.8733362
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.00
Output dim: 0, lower bound: -0.8733366, upper bound: 0.8726909

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8440590, 1.8443944
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7915926, 2.7920837
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3577261, 2.3579614
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6013870, 2.6013865
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7619915, 2.7598598
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0473876, 2.0477066
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2827492, 2.2830470
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1824570, 3.1791835
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0481467, 2.0490048
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3085880, 2.3069739

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 552

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8720095, upper bound: 0.8728499
time: 21.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8722042, upper bound: 0.8726552
time: 7.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8442516, 1.8440585
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7918749, 2.7915931
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3578606, 2.3577263
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6013861, 2.6013861
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7598591, 2.7610815
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0475702, 2.0473876
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2829189, 2.2827489
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1791840, 3.1810575
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0486374, 2.0481462
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3069739, 2.3078990

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727869, upper bound: 0.8715692
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8722134, upper bound: 0.8721417
time: 18.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 41.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 41.09
Output dim: 0, lower bound: -0.8720095, upper bound: 0.8728499
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 41.09
Output dim: 0, lower bound: -0.8722042, upper bound: 0.8726552
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 41.09
Output dim: 0, lower bound: -0.8727869, upper bound: 0.8715692
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 41.09
Output dim: 0, lower bound: -0.8722134, upper bound: 0.8721417

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8440542, 1.8451447
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7915850, 2.7936039
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3526702, 2.3543386
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5993829, 2.5985885
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7613411, 2.7593932
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0447335, 2.0440030
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2808108, 2.2803426
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1839533, 3.1791778
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0478263, 2.0485580
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3059144, 2.3050587

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8714547, upper bound: 0.8717262
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8708848, upper bound: 0.8722982
time: 11.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8448091, 1.8443904
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7931128, 2.7920752
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3541036, 2.3529053
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5985894, 2.5993819
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7615252, 2.7592089
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0436840, 2.0450525
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2800450, 2.2811084
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1824512, 3.1806803
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0476995, 2.0486844
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3066726, 2.3043008

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8691410, upper bound: 0.8715191
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8710712, upper bound: 0.8695900
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8442416, 1.8440571
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7915659, 2.7916574
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3580384, 2.3568780
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6008348, 2.6015024
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7598782, 2.7609866
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0469894, 2.0475094
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2825727, 2.2828188
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1791496, 3.1810641
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0486579, 2.0480285
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3070440, 2.3075693

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 453

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727657, upper bound: 0.8715699
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727854, upper bound: 0.8715500
time: 8.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 30.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 30.83
Output dim: 0, lower bound: -0.8714547, upper bound: 0.8717262
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 30.83
Output dim: 0, lower bound: -0.8708848, upper bound: 0.8722982
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 30.83
Output dim: 0, lower bound: -0.8691410, upper bound: 0.8715191
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 30.83
Output dim: 0, lower bound: -0.8710712, upper bound: 0.8695900
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -0.8727657, upper bound: 0.8715699
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.83
Output dim: 0, lower bound: -0.8727854, upper bound: 0.8715500

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8441765, 1.8438661
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7903066, 2.7879176
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3521199, 2.3548629
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5984316, 2.5944009
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7494693, 2.7574565
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0461197, 2.0449564
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2819738, 2.2810528
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1783986, 3.1788602
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0481625, 2.0465684
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3059411, 2.3043246

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8681651, upper bound: 0.8669339
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8681651, upper bound: 0.8669339
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8440506, 1.8439918
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7878270, 2.7903972
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3560233, 2.3509598
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5937328, 2.5990992
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7563491, 2.7505772
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0444365, 2.0466399
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2808075, 2.2822192
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1769452, 3.1803136
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0471978, 2.0475330
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3037992, 2.3064666

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698650, upper bound: 0.8715441
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727815, upper bound: 0.8686450
time: 10.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 33.68 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 0, lower bound: -0.8681651, upper bound: 0.8669339
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 0, lower bound: -0.8681651, upper bound: 0.8669339
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 33.68
Output dim: 0, lower bound: -0.8698650, upper bound: 0.8715441
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.68
Output dim: 0, lower bound: -0.8727815, upper bound: 0.8686450

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8385053, 1.8376536
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7855978, 2.7832422
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3372254, 2.3345120
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6051922, 2.6086545
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7503963, 2.7437742
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0480280, 2.0487421
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2686720, 2.2683513
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1755953, 3.1791315
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0276713, 2.0252213
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2870584, 2.2873344

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 6219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727774, upper bound: 0.8641768
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8683055, upper bound: 0.8686384
time: 7.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 30.42 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 0, lower bound: -0.8727774, upper bound: 0.8641768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 30.42
Output dim: 0, lower bound: -0.8683055, upper bound: 0.8686384

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8355503, 1.8335536
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7837095, 2.7806215
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3311620, 2.3301442
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6044836, 2.6081438
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7363715, 2.7242918
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0441704, 2.0459619
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2521257, 2.2564361
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1691651, 3.1702094
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0274796, 2.0249553
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2711997, 2.2653117

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8713353, upper bound: 0.8633783
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727646, upper bound: 0.8633765
time: 11.06 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 30.71 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 30.71
Output dim: 0, lower bound: -0.8713353, upper bound: 0.8633783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 30.71
Output dim: 0, lower bound: -0.8727646, upper bound: 0.8633765

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8337121, 1.8303154
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7805586, 2.7750678
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3305531, 2.3290725
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6039400, 2.6071863
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7138615, 2.7115235
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0422468, 2.0425713
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2495832, 2.2519524
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1407261, 3.1540790
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0221682, 2.0155840
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2568469, 2.2571745

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8681456, upper bound: 0.8586819
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8681456, upper bound: 0.8586819
time: 6.85 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 28.03 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 28.03
Output dim: 0, lower bound: -0.8681456, upper bound: 0.8586819
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 28.03
Output dim: 0, lower bound: -0.8681456, upper bound: 0.8586819
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8442518711090088
rel_dist={0: [-0.8742930134486402, 0.8742908316971416]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2012.41 seconds
