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
execution time: IAR + LP analysis = 14.08 + 33.53 = 47.60 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.40 seconds, max iter: 100)

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
Binary search time: 218.72 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3333.68 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945762, upper bound: 1.2920582
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920580, upper bound: 1.2945761
time: 5.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.03
Output dim: 0, lower bound: -1.2945762, upper bound: 1.2920582
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.03
Output dim: 0, lower bound: -1.2920580, upper bound: 1.2945761

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0885463, 2.0889344
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1767893, 3.1789670
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6455698, 2.6389999
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8941965, 2.9015169
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0910726, 3.0889196
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2255130, 2.2306776
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4571438, 2.4586601
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4984541, 3.5008874
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2375059, 2.2363520
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.5058460, 2.5031812

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2894335, upper bound: 1.2920500
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945681, upper bound: 1.2869183
time: 5.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0889344, 2.0885463
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1789675, 3.1767893
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6389999, 2.6455698
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9015169, 2.8941965
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0889192, 3.0910721
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2306776, 2.2255132
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4586601, 2.4571445
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.5008879, 3.4984541
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2363520, 2.2375057
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.5031815, 2.5058455

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2869181, upper bound: 1.2945677
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920499, upper bound: 1.2894335
time: 5.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.90
Output dim: 0, lower bound: -1.2894335, upper bound: 1.2920500
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.90
Output dim: 0, lower bound: -1.2945681, upper bound: 1.2869183
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.90
Output dim: 0, lower bound: -1.2869181, upper bound: 1.2945677
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.90
Output dim: 0, lower bound: -1.2920499, upper bound: 1.2894335

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0822082, 2.0844450
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1696339, 3.1833043
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6322579, 2.6202037
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9037523, 2.9155145
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0842695, 3.0840993
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2276154, 2.2362559
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4432769, 2.4488344
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4974971, 3.4995384
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2151942, 2.2205381
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4867134, 2.4896293

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2894235, upper bound: 1.2829676
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2801247, upper bound: 1.2920421
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0840569, 2.0825965
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1811266, 3.1718111
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6267734, 2.6256876
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9081945, 2.9110727
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0862513, 3.0821166
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2310915, 2.2327795
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4473186, 2.4447925
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4971061, 3.4999309
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2216921, 2.2140403
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4922934, 2.4840493

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945580, upper bound: 1.2778650
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852351, upper bound: 1.2869083
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0825963, 2.0840569
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1718111, 3.1811266
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6256876, 2.6267736
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9110727, 2.9081941
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0821161, 3.0862517
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2327795, 2.2310915
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4447923, 2.4473188
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4999309, 3.4971051
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2140408, 2.2216918
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4840493, 2.4922936

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2869085, upper bound: 1.2852346
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2778656, upper bound: 1.2945599
time: 7.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0844450, 2.0822084
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1833048, 3.1696334
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6202035, 2.6322577
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9155149, 2.9037519
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0840998, 3.0842690
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2362556, 2.2276154
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4488349, 2.4432769
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4995379, 3.4974980
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2205381, 2.2151940
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4896293, 2.4867136

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920403, upper bound: 1.2801244
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829680, upper bound: 1.2894231
time: 12.22 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 31.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2894235, upper bound: 1.2829676
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2801247, upper bound: 1.2920421
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2945580, upper bound: 1.2778650
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2852351, upper bound: 1.2869083
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2869085, upper bound: 1.2852346
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2778656, upper bound: 1.2945599
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2920403, upper bound: 1.2801244
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.30
Output dim: 0, lower bound: -1.2829680, upper bound: 1.2894231

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0807810, 2.0803452
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1687222, 3.1806841
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6261935, 2.6180971
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9030428, 2.9152684
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0775213, 3.0646172
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2237568, 2.2349124
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4267316, 2.4430928
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4943933, 3.4906158
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2151008, 2.2202706
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4790740, 2.4676061

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2894224, upper bound: 1.2827542
time: 18.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2891678, upper bound: 1.2829670
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0781088, 2.0830173
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1670132, 3.1823921
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6301503, 2.6141393
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9035053, 2.9148054
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0647879, 3.0773520
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2262716, 2.2323978
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4375348, 2.4322889
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4885759, 3.4964333
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2149267, 2.2204447
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4646907, 2.4819889

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2801236, upper bound: 1.2917873
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2799009, upper bound: 1.2920391
time: 6.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0826292, 2.0784967
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1802139, 3.1691909
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6207094, 2.6235812
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9074850, 2.9108262
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0795050, 3.0626345
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2272329, 2.2314360
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4307733, 2.4390512
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4940004, 3.4910083
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2215986, 2.2137728
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4846539, 2.4620261

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2945569, upper bound: 1.2776541
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2943041, upper bound: 1.2778646
time: 7.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0799570, 2.0811689
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1785069, 3.1708994
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6246662, 2.6196232
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9079475, 2.9103632
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0667696, 3.0753694
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2297478, 2.2289214
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4415765, 2.4282470
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4881830, 3.4968262
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2214246, 2.2139468
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4702706, 2.4764090

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2852341, upper bound: 1.2866525
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2850055, upper bound: 1.2869075
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0811691, 2.0799570
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1709003, 3.1785064
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6196232, 2.6246660
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9103632, 2.9079480
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0753698, 3.0667696
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2289209, 2.2297480
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4282470, 2.4415762
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4968271, 3.4881825
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2139473, 2.2214243
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4764090, 2.4702704

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2869074, upper bound: 1.2850052
time: 12.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2866528, upper bound: 1.2852337
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0784969, 2.0826292
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1691914, 3.1802149
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6235809, 2.6207092
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9108257, 2.9074850
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0626345, 3.0795040
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2314358, 2.2272334
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4390512, 2.4307730
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4910078, 3.4940004
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2137728, 2.2215984
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4620261, 2.4846539

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2778646, upper bound: 1.2943043
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2776523, upper bound: 1.2945573
time: 7.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0830173, 2.0781085
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1823921, 3.1670132
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6141396, 2.6301501
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9148054, 2.9035058
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0773516, 3.0647869
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2323980, 2.2262719
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4322886, 2.4375343
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4964342, 3.4885755
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2204447, 2.2149265
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4819889, 2.4646904

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2920392, upper bound: 1.2799009
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2917867, upper bound: 1.2801237
time: 8.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0803452, 2.0807807
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1806850, 3.1687217
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6180973, 2.6261933
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9152679, 2.9030428
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0646181, 3.0775213
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2349119, 2.2237570
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4430928, 2.4267313
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4906168, 3.4943929
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2202711, 2.2151005
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4676061, 2.4790740

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2829670, upper bound: 1.2891679
time: 10.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2827545, upper bound: 1.2894221
time: 10.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 33.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2894224, upper bound: 1.2827542
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2891678, upper bound: 1.2829670
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2801236, upper bound: 1.2917873
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2799009, upper bound: 1.2920391
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2945569, upper bound: 1.2776541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2943041, upper bound: 1.2778646
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2852341, upper bound: 1.2866525
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2850055, upper bound: 1.2869075
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2869074, upper bound: 1.2850052
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2866528, upper bound: 1.2852337
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2778646, upper bound: 1.2943043
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2776523, upper bound: 1.2945573
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2920392, upper bound: 1.2799009
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2917867, upper bound: 1.2801237
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2829670, upper bound: 1.2891679
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.79
Output dim: 0, lower bound: -1.2827545, upper bound: 1.2894221

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0807943, 2.0798779
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1689930, 3.1722040
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6262007, 2.6177146
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9033017, 2.9073000
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0745649, 3.0647073
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2238503, 2.2320743
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4261222, 2.4431133
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4865522, 3.4908848
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2150593, 2.2202761
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4738922, 2.4677675

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2873738, upper bound: 1.2820102
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2862095, upper bound: 1.2820150
time: 9.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0803137, 2.0803452
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1602421, 3.1806841
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6258106, 2.6180971
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8950734, 2.9152684
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0775213, 3.0616608
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2209187, 2.2349124
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4267316, 2.4424829
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4943933, 3.4827743
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2151008, 2.2202296
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4790740, 2.4624243

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2871193, upper bound: 1.2822364
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2859550, upper bound: 1.2822396
time: 9.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0781221, 2.0825500
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1672840, 3.1739125
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6301575, 2.6137569
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9037652, 2.9068370
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0618296, 3.0774417
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2263651, 2.2295597
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4369245, 2.4323092
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4807339, 3.4967027
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2148857, 2.2204502
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4595089, 2.4821503

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2781221, upper bound: 1.2911010
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2771167, upper bound: 1.2911088
time: 13.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0776415, 2.0830173
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1585340, 3.1823921
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6297674, 2.6141393
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8955369, 2.9148054
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0647879, 3.0743952
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2234335, 2.2323978
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4375348, 2.4316790
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4885759, 3.4885921
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2149267, 2.2204034
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4646907, 2.4768071

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2779018, upper bound: 1.2913540
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2768970, upper bound: 1.2913613
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0826430, 2.0780294
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1804857, 3.1607108
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6207166, 2.6231987
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9077439, 2.9028578
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0765467, 3.0627246
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2273264, 2.2285979
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4301639, 2.4390714
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4861593, 3.4912777
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2215571, 2.2137783
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4794722, 2.4621875

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2925104, upper bound: 1.2769018
time: 7.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2913498, upper bound: 1.2769066
time: 8.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0821619, 2.0784967
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1717348, 3.1691909
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6203270, 2.6235812
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8995156, 2.9108262
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0795050, 3.0596781
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2243948, 2.2314360
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4307733, 2.4384413
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4940004, 3.4831672
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2215986, 2.2137318
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4846539, 2.4568443

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2922586, upper bound: 1.2771217
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2910959, upper bound: 1.2771265
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0799708, 2.0807016
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1787777, 3.1624193
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6246734, 2.6192408
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9082074, 2.9023948
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0638132, 3.0754590
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2298412, 2.2260833
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4409671, 2.4282675
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4803419, 3.4970956
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2213831, 2.2139523
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4650888, 2.4765701

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2832370, upper bound: 1.2859605
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2822297, upper bound: 1.2859688
time: 11.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0794897, 2.0811689
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1700268, 3.1708994
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6242838, 2.6196232
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.8999791, 2.9103632
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0667696, 3.0724125
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2269096, 2.2289214
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4415765, 2.4276371
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4881830, 3.4889851
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2214246, 2.2139056
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4702706, 2.4712272

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2830093, upper bound: 1.2862140
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2820035, upper bound: 1.2862222
time: 8.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0811825, 2.0794897
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1711702, 3.1700263
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6196308, 2.6242836
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9106221, 2.8999796
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0724115, 3.0668597
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2290144, 2.2269099
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4276376, 2.4415965
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4889851, 3.4884520
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2139058, 2.2214298
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4712272, 2.4704318

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2862226, upper bound: 1.2820030
time: 13.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2862148, upper bound: 1.2830088
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0807018, 2.0799570
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1624193, 3.1785064
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6192408, 2.6246660
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9023957, 2.9079480
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0753698, 3.0638127
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2260828, 2.2297480
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4282470, 2.4409664
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4968271, 3.4803414
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2139473, 2.2213831
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4764090, 2.4650886

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2859681, upper bound: 1.2822316
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2859602, upper bound: 1.2832364
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0785103, 2.0821619
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1694622, 3.1717348
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6235886, 2.6203268
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9110856, 2.8995166
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0596781, 3.0795941
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2315292, 2.2243953
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4384418, 2.4307935
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4831667, 3.4942698
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2137318, 2.2216039
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4568443, 2.4848151

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2771263, upper bound: 1.2910954
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2771215, upper bound: 1.2922589
time: 9.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -2.0780296, 2.0826292
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -3.1607113, 3.1802149
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.6231985, 2.6207092
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.9028573, 2.9074850
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -3.0626345, 3.0765476
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.2285976, 2.2272334
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.4390512, 2.4301631
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.4910078, 3.4861593
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.2137728, 2.2215571
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.4620261, 2.4794722

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2769066, upper bound: 1.2913510
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.2769018, upper bound: 1.2925097
time: 5.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2873738, upper bound: 1.2820102
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2862095, upper bound: 1.2820150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2871193, upper bound: 1.2822364
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2859550, upper bound: 1.2822396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2781221, upper bound: 1.2911010
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2771167, upper bound: 1.2911088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2779018, upper bound: 1.2913540
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2768970, upper bound: 1.2913613
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2925104, upper bound: 1.2769018
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2913498, upper bound: 1.2769066
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2922586, upper bound: 1.2771217
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2910959, upper bound: 1.2771265
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2832370, upper bound: 1.2859605
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2822297, upper bound: 1.2859688
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2830093, upper bound: 1.2862140
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2820035, upper bound: 1.2862222
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2862226, upper bound: 1.2820030
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2862148, upper bound: 1.2830088
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2859681, upper bound: 1.2822316
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2859602, upper bound: 1.2832364
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2771263, upper bound: 1.2910954
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2771215, upper bound: 1.2922589
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2769066, upper bound: 1.2913510
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.35
Output dim: 0, lower bound: -1.2769018, upper bound: 1.2925097
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.35
Output dim: 0, lower bound: -1.2920392, upper bound: 1.2799009
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.35
Output dim: 0, lower bound: -1.2917867, upper bound: 1.2801237
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.35
Output dim: 0, lower bound: -1.2829670, upper bound: 1.2891679
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.35
Output dim: 0, lower bound: -1.2827545, upper bound: 1.2894221
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.088984727859497
rel_dist={0: [-1.2945795095372876, 1.2945797204952658]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840167, upper bound: 0.9827040
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9827034, upper bound: 0.9840178
time: 7.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.78
Output dim: 0, lower bound: -0.9840167, upper bound: 0.9827040
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.78
Output dim: 0, lower bound: -0.9827034, upper bound: 0.9840178

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9049964, 1.9052186
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8862715, 2.8875155
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4263744, 2.4226201
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6684332, 2.6726160
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8424602, 2.8412304
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0877113, 2.0906625
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3251972, 2.3260634
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2583609, 3.2597508
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0952463, 2.0945868
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3560004, 2.3544779

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9810006, upper bound: 0.9826996
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840126, upper bound: 0.9796862
time: 15.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.9052186, 1.9049966
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8875160, 2.8862710
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4226198, 2.4263744
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6726160, 2.6684332
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8412309, 2.8424602
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0906625, 2.0877113
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3260632, 2.3251975
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2597513, 3.2583609
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0945868, 2.0952458
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3544779, 2.3560002

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9796859, upper bound: 0.9840127
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9826992, upper bound: 0.9810030
time: 7.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.96
Output dim: 0, lower bound: -0.9810006, upper bound: 0.9826996
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.96
Output dim: 0, lower bound: -0.9840126, upper bound: 0.9796862
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.96
Output dim: 0, lower bound: -0.9796859, upper bound: 0.9840127
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.96
Output dim: 0, lower bound: -0.9826992, upper bound: 0.9810030

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8986588, 1.8999369
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8791161, 2.8869271
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4107118, 2.4038239
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6779890, 2.6847095
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8356571, 2.8355603
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0898132, 2.0947509
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3113303, 2.3145056
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2572360, 3.2584019
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0729342, 2.0759881
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3368683, 2.3385344

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809951, upper bound: 0.9765230
time: 10.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9748711, upper bound: 0.9826942
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8997149, 1.8988807
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8856831, 2.8803601
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4075780, 2.4069576
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6805277, 2.6821718
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8367901, 2.8344274
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0917997, 2.0927644
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3136392, 2.3121960
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2570109, 3.2586265
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0766473, 2.0722752
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3400569, 2.3353460

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840071, upper bound: 0.9735011
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778926, upper bound: 0.9796799
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8988805, 1.8997152
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8803606, 2.8856831
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4069576, 2.4075782
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6821718, 2.6805267
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8344278, 2.8367903
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0927644, 2.0918000
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3121963, 2.3136394
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2586265, 3.2570119
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0722752, 2.0766473
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3353462, 2.3400569

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9796803, upper bound: 0.9778933
time: 9.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9735014, upper bound: 0.9840067
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8999372, 1.8986588
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8869276, 2.8791156
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4038239, 2.4107120
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6847105, 2.6779881
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8355608, 2.8356574
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0947509, 2.0898135
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3145061, 2.3113298
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2584014, 3.2572360
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0759883, 2.0729342
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3385344, 2.3368683

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9826937, upper bound: 0.9748711
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765229, upper bound: 0.9809977
time: 7.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9809951, upper bound: 0.9765230
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9748711, upper bound: 0.9826942
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9840071, upper bound: 0.9735011
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9778926, upper bound: 0.9796799
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9796803, upper bound: 0.9778933
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9735014, upper bound: 0.9840067
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9826937, upper bound: 0.9748711
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.94
Output dim: 0, lower bound: -0.9765229, upper bound: 0.9809977

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8960857, 1.8958371
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8774719, 2.8843074
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4046478, 2.4000211
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6772795, 2.6842651
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8234525, 2.8160782
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0859547, 2.0923297
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2947841, 2.3041337
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2516375, 3.2494793
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0727663, 2.0757205
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3230643, 2.3165112

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809940, upper bound: 0.9765223
time: 13.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9809940, upper bound: 0.9765217
time: 15.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8945589, 1.8973641
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8764954, 2.8852835
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4069090, 2.3977594
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6775436, 2.6840010
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8161750, 2.8233552
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0873919, 2.0908928
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3009582, 2.2979600
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2483130, 3.2528038
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0726671, 2.0758200
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3148451, 2.3247299

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9748700, upper bound: 0.9826932
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9748700, upper bound: 0.9826955
time: 7.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8971424, 1.8947809
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8840389, 2.8777399
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4015141, 2.4031549
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6798172, 2.6817269
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8245854, 2.8149452
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0879412, 2.0903432
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2970939, 2.3018241
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2514143, 3.2497039
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0764794, 2.0720077
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3262529, 2.3133228

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840060, upper bound: 0.9735000
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9840060, upper bound: 0.9735001
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8956156, 1.8963077
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8830624, 2.8787160
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4037752, 2.4008932
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6800823, 2.6814623
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8173079, 2.8222222
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0893784, 2.0889063
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3032670, 2.2956502
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2480898, 3.2530284
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0763798, 2.0721068
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3180337, 2.3215413

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778915, upper bound: 0.9796792
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9778915, upper bound: 0.9796792
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8963079, 1.8956153
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8787155, 2.8830633
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4008932, 2.4037750
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6814623, 2.6800823
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8222222, 2.8173082
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0889063, 2.0893786
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2956500, 2.3032670
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2530279, 3.2480893
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0721073, 2.0763798
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3215413, 2.3180337

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9796793, upper bound: 0.9778941
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9796792, upper bound: 0.9778941
time: 8.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8947811, 1.8971422
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8777409, 2.8840389
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4031553, 2.4015138
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6817265, 2.6798177
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8149457, 2.8245850
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0903435, 2.0879416
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3018241, 2.2970939
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2497034, 3.2514133
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0720077, 2.0764792
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3133225, 2.3262529

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9735003, upper bound: 0.9840058
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9735003, upper bound: 0.9840058
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8973641, 1.8945589
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8852844, 2.8764954
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3977594, 2.4069085
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6840010, 2.6775436
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8233552, 2.8161752
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0908928, 2.0873921
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2979598, 2.3009574
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2528048, 3.2483134
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0758200, 2.0726666
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3247299, 2.3148451

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9826927, upper bound: 0.9748707
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9826926, upper bound: 0.9748709
time: 6.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8958368, 1.8960860
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8843079, 2.8774719
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4000216, 2.4046476
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6842651, 2.6772795
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8160787, 2.8234520
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0923300, 2.0859551
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3041339, 2.2947843
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2494802, 3.2516379
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0757208, 2.0727663
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3165112, 2.3230643

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765219, upper bound: 0.9809938
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9765218, upper bound: 0.9809938
time: 5.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9809940, upper bound: 0.9765223
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9809940, upper bound: 0.9765217
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9748700, upper bound: 0.9826932
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9748700, upper bound: 0.9826955
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9840060, upper bound: 0.9735000
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9840060, upper bound: 0.9735001
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9778915, upper bound: 0.9796792
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9778915, upper bound: 0.9796792
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9796793, upper bound: 0.9778941
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9796792, upper bound: 0.9778941
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9735003, upper bound: 0.9840058
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9735003, upper bound: 0.9840058
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9826927, upper bound: 0.9748707
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9826926, upper bound: 0.9748709
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9765219, upper bound: 0.9809938
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.61
Output dim: 0, lower bound: -0.9765218, upper bound: 0.9809938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8958936, 1.8953698
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8739929, 2.8758273
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4044876, 2.3996387
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6740117, 2.6762967
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8204951, 2.8148625
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0847921, 2.0894916
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2941747, 2.3038840
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2437963, 3.2462726
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0727253, 2.0757060
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3178825, 2.3143826

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9800179, upper bound: 0.9758450
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9789526, upper bound: 0.9758479
time: 5.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8956184, 1.8956447
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8689919, 2.8808279
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4042654, 2.3998613
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6693101, 2.6809988
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8222365, 2.8131216
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0831165, 2.0911667
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2945342, 2.3035238
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2484312, 3.2416382
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0727515, 2.0756793
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3209357, 2.3113294

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9800178, upper bound: 0.9758451
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9789526, upper bound: 0.9758479
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8943663, 1.8968968
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8730164, 2.8768039
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4067488, 2.3973770
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6742768, 2.6760325
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8132186, 2.8221393
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0862293, 2.0880547
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3003478, 2.2977102
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2404718, 3.2495971
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0726256, 2.0758054
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3096633, 2.3226013

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9739257, upper bound: 0.9819671
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9728220, upper bound: 0.9819725
time: 13.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8940916, 1.8971717
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8680153, 2.8818040
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4065261, 2.3975997
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6695752, 2.6807342
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8149600, 2.8203986
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0845537, 2.0897298
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3007083, 2.2973502
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2451067, 3.2449627
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0726523, 2.0757787
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3127165, 2.3195481

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9739256, upper bound: 0.9819680
time: 9.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9728219, upper bound: 0.9819726
time: 8.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8969498, 1.8943136
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8805599, 2.8692598
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4013538, 2.4027724
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6765504, 2.6737585
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8216281, 2.8137295
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0867786, 2.0875051
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2964845, 2.3015745
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2435722, 3.2464972
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0764380, 2.0719929
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3210711, 2.3111942

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9830309, upper bound: 0.9728234
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9819654, upper bound: 0.9728266
time: 5.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8966751, 1.8945885
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8755598, 2.8742604
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4011316, 2.4029951
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6718488, 2.6784606
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8233695, 2.8119888
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0851030, 2.0891802
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2968440, 2.3012142
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2482071, 3.2418623
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0764647, 2.0719662
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3241243, 2.3081410

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9830308, upper bound: 0.9728236
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9819653, upper bound: 0.9728266
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8954229, 1.8958404
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8795834, 2.8702359
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4036150, 2.4005108
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6768155, 2.6734939
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8143516, 2.8210065
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0882158, 2.0860682
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3026576, 2.2954006
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2402477, 3.2498217
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0763388, 2.0720923
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3128519, 2.3194127

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9769473, upper bound: 0.9789547
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9758436, upper bound: 0.9789598
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8951483, 1.8961155
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8745832, 2.8752365
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4033928, 2.4007335
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6721139, 2.6781960
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8160930, 2.8192656
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0865402, 2.0877435
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3030181, 2.2950404
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2448826, 3.2451868
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0763655, 2.0720658
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3159051, 2.3163595

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9769473, upper bound: 0.9789547
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9758435, upper bound: 0.9789598
time: 5.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8961153, 1.8951480
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8752365, 2.8745828
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4007335, 2.4033926
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6781964, 2.6721139
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8192658, 2.8160925
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0877428, 2.0865405
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2950406, 2.3030174
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2451868, 3.2448826
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0720658, 2.0763652
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3163595, 2.3159051

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9789600, upper bound: 0.9758437
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9789549, upper bound: 0.9769466
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8958406, 1.8954229
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8702364, 2.8795838
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4005108, 2.4036152
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6734948, 2.6768155
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8210063, 2.8143516
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0860682, 2.0882158
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2954011, 2.3026571
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2498217, 3.2402477
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0720925, 2.0763385
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3194127, 2.3128519

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9789599, upper bound: 0.9758437
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9789549, upper bound: 0.9769466
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8945885, 1.8966749
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8742609, 2.8755593
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4029951, 2.4011314
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6784596, 2.6718493
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8119884, 2.8233693
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0891800, 2.0851035
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3012147, 2.2968442
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2418623, 3.2482071
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0719662, 2.0764647
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3081408, 2.3241241

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9728266, upper bound: 0.9819648
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9728236, upper bound: 0.9830304
time: 9.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8943138, 1.8969500
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8692608, 2.8805594
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.4027724, 2.4013541
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6737580, 2.6765513
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8137298, 2.8216283
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0875053, 2.0867789
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.3015742, 2.2964840
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2464972, 3.2435722
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0719929, 2.0764380
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3111944, 2.3210711

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9728266, upper bound: 0.9819649
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9728236, upper bound: 0.9830307
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8971720, 1.8940916
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8818045, 2.8680158
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3975997, 2.4065261
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6807332, 2.6695752
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8203988, 2.8149595
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0897293, 2.0845540
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2973504, 2.3007076
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2449627, 3.2451067
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0757790, 2.0726521
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3195481, 2.3127165

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9819727, upper bound: 0.9728247
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9819676, upper bound: 0.9739283
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8968968, 1.8943665
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.8768044, 2.8730159
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3973770, 2.4067488
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6760316, 2.6742773
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.8221393, 2.8132186
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0880547, 2.0862293
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2977099, 2.3003476
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.2495966, 3.2404723
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0758057, 2.0726256
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3226013, 2.3096633

Time for backsubstitution: 12.52 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.9054350852966309
rel_dist={0: [-0.9840186515659699, 0.9840187223222951]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5778
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5778

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742885, upper bound: 0.8733165
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733167, upper bound: 0.8742894
time: 5.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.84 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.84
Output dim: 0, lower bound: -0.8742885, upper bound: 0.8733165
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.84
Output dim: 0, lower bound: -0.8733167, upper bound: 0.8742894

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8438134, 1.8439798
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7894325, 2.7903652
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3533092, 2.3504934
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5931787, 2.5963159
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7595897, 2.7586672
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0417776, 2.0439906
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2812147, 2.2818646
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1783295, 3.1793718
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0478263, 2.0473316
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3060522, 2.3049099

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8713674, upper bound: 0.8733133
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742847, upper bound: 0.8703952
time: 5.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8439798, 1.8438134
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7903652, 2.7894320
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3504934, 2.3533092
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5963163, 2.5931787
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7586675, 2.7595894
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0439906, 2.0417776
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2818642, 2.2812150
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1793718, 3.1783295
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0473318, 2.0478261
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.3049102, 2.3060517

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5773
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5773

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8703955, upper bound: 0.8742850
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733128, upper bound: 0.8713683
time: 5.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 0, lower bound: -0.8713674, upper bound: 0.8733133
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 0, lower bound: -0.8742847, upper bound: 0.8703952
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 0, lower bound: -0.8703955, upper bound: 0.8742850
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.03
Output dim: 0, lower bound: -0.8733128, upper bound: 0.8713683

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8374753, 1.8384342
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7822771, 2.7881351
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3368630, 2.3316972
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6027346, 2.6077747
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7527866, 2.7527142
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0438795, 2.0475826
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2673469, 2.2697291
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1771483, 3.1780229
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0255146, 2.0278049
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2869196, 2.2881694

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8713635, upper bound: 0.8687358
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8669564, upper bound: 0.8733092
time: 8.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8382678, 1.8376420
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7872019, 2.7832098
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3345132, 2.3340476
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6046381, 2.6058712
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7536364, 2.7518644
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0453691, 2.0460927
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2690797, 2.2679970
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1769805, 3.1781917
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0282993, 2.0250199
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2893114, 2.2857780

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742808, upper bound: 0.8659139
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8698103, upper bound: 0.8703923
time: 12.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8376422, 1.8382678
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7832098, 2.7872019
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3340478, 2.3345129
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6058722, 2.6046376
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7518644, 2.7536364
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0460930, 2.0453694
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2679973, 2.2690797
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1781917, 3.1769805
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0250201, 2.0282991
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2857780, 2.2893114

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8703916, upper bound: 0.8698100
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659142, upper bound: 0.8742817
time: 18.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8384342, 1.8374755
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7881355, 2.7822766
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3316970, 2.3368633
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6077757, 2.6027336
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7527142, 2.7527866
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0475826, 2.0438795
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2697291, 2.2673473
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1780238, 3.1771488
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0278049, 2.0255144
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2881694, 2.2869198

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733089, upper bound: 0.8669573
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8687356, upper bound: 0.8713641
time: 10.42 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 29.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8713635, upper bound: 0.8687358
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8669564, upper bound: 0.8733092
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8742808, upper bound: 0.8659139
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8698103, upper bound: 0.8703923
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8703916, upper bound: 0.8698100
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8659142, upper bound: 0.8742817
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8733089, upper bound: 0.8669573
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 29.03
Output dim: 0, lower bound: -0.8687356, upper bound: 0.8713641

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8333759, 1.8354797
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7796574, 2.7862473
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3324947, 2.3256328
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6022234, 2.6070662
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7333045, 2.7386894
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0410986, 2.0437243
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2554317, 2.2531836
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1682272, 3.1715941
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0252471, 2.0276117
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2648969, 2.2723103

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8669525, upper bound: 0.8733084
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8669548, upper bound: 0.8733085
time: 8.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8353133, 1.8335421
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7853146, 2.7805896
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3284488, 2.3296795
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6039286, 2.6053605
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7396121, 2.7323823
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0415106, 2.0433123
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2525344, 2.2560818
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1705503, 3.1692691
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0281062, 2.0247524
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2734523, 2.2637548

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742781, upper bound: 0.8659140
time: 9.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8742793, upper bound: 0.8659140
time: 7.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8335419, 1.8353133
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7805901, 2.7853141
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3296795, 2.3284485
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6053600, 2.6039286
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7323823, 2.7396119
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0433121, 2.0415111
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2560821, 2.2525342
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1692686, 3.1705508
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0247526, 2.0281062
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2637548, 2.2734525

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742799
time: 10.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8659164, upper bound: 0.8742771
time: 6.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8354797, 1.8333757
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7862473, 2.7796564
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3256330, 2.3324947
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6070662, 2.6022234
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7386899, 2.7333045
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0437241, 2.0410991
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2531838, 2.2554317
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1715937, 3.1682262
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0276117, 2.0252469
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2723103, 2.2648966

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6219
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733055, upper bound: 0.8669550
time: 13.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8733076, upper bound: 0.8669519
time: 6.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 33.67 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8669525, upper bound: 0.8733084
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8669548, upper bound: 0.8733085
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8742781, upper bound: 0.8659140
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8742793, upper bound: 0.8659140
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8659135, upper bound: 0.8742799
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8659164, upper bound: 0.8742771
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8733055, upper bound: 0.8669550
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.67
Output dim: 0, lower bound: -0.8733076, upper bound: 0.8669519

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8331146, 1.8350124
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7749271, 2.7777677
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3322792, 2.3252504
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5977807, 2.5990977
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7303481, 2.7370386
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0395174, 2.0408862
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2548223, 2.2528439
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1603851, 3.1672287
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0252056, 2.0275905
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2597151, 2.2694185

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8663646, upper bound: 0.8726994
time: 14.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727039
time: 10.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8329086, 1.8352184
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7711763, 2.7815175
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3321123, 2.3254175
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5942540, 2.6026239
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7316537, 2.7357330
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0382605, 2.0421426
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2550921, 2.2525737
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1638613, 3.1637526
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0252256, 2.0275705
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2620049, 2.2671285

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8663680, upper bound: 0.8726979
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727033
time: 7.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8350525, 1.8330748
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7805843, 2.7721095
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3282332, 2.3292971
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5994859, 2.5973921
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7366557, 2.7307312
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0399294, 2.0404742
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2519240, 2.2557421
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1627092, 3.1649036
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0280652, 2.0247312
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2682705, 2.2608631

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8737233, upper bound: 0.8652882
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726944, upper bound: 0.8652867
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8348460, 1.8332810
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7768345, 2.7758598
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3280659, 2.3294640
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5959592, 2.6009183
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7379613, 2.7294257
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0386724, 2.0417306
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2521949, 2.2554719
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1661863, 3.1614275
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0280848, 2.0247111
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2705603, 2.2585731

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8737239, upper bound: 0.8652849
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726967, upper bound: 0.8652872
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8332810, 1.8348460
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7758598, 2.7768340
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3294640, 2.3280661
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6009183, 2.5959601
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7294259, 2.7379611
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0417299, 2.0386729
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2554717, 2.2521944
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1614275, 3.1661859
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0247111, 2.0280848
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2585731, 2.2705605

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726974
time: 14.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737236
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8330746, 1.8350523
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7721100, 2.7805843
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3292971, 2.3282332
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5973916, 2.5994868
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7307315, 2.7366552
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0404739, 2.0399294
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2557425, 2.2519243
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1649036, 3.1627097
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0247312, 2.0280650
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2608633, 2.2682707

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726947
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737243
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8352184, 1.8329084
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7815180, 2.7711763
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3254175, 2.3321123
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.6026235, 2.5942550
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7357326, 2.7316537
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0421429, 2.0382609
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2525744, 2.2550919
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1637526, 3.1638608
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0275707, 2.0252256
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2671285, 2.2620049

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727003, upper bound: 0.8652832
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726964, upper bound: 0.8663674
time: 6.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8350124, 1.8331146
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7777672, 2.7749267
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3252506, 2.3322794
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5990968, 2.5977812
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7370391, 2.7303479
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0408859, 2.0395174
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2528443, 2.2548218
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1672287, 3.1603851
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0275908, 2.0252056
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2694187, 2.2597148

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8727026, upper bound: 0.8652845
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8726987, upper bound: 0.8663651
time: 14.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 36.55 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8663646, upper bound: 0.8726994
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727039
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8663680, upper bound: 0.8726979
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8652840, upper bound: 0.8727033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8737233, upper bound: 0.8652882
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8726944, upper bound: 0.8652867
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8737239, upper bound: 0.8652849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8726967, upper bound: 0.8652872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737243
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8727003, upper bound: 0.8652832
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8726964, upper bound: 0.8663674
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8727026, upper bound: 0.8652845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 36.55
Output dim: 0, lower bound: -0.8726987, upper bound: 0.8663651

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8331146, 1.8350248
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7746305, 2.7778416
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3324561, 2.3244021
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5972314, 2.5992150
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7303677, 2.7369428
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0389357, 2.0410073
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2544780, 2.2529161
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1603508, 3.1672359
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0252299, 2.0274720
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2597837, 2.2690890

Time for backsubstitution: 12.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8652914, upper bound: 0.8715617
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8652928, upper bound: 0.8687307
time: 8.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8331146, 1.8350127
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7749271, 2.7774706
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3314314, 2.3252504
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5977807, 2.5985470
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7302523, 2.7370386
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0395174, 2.0403049
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2548223, 2.2524996
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1603851, 3.1671944
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0250874, 2.0275905
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2593856, 2.2694185

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8642016, upper bound: 0.8715648
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8642029, upper bound: 0.8687342
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8329086, 1.8352311
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7708807, 2.7815919
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3322892, 2.3245692
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5937047, 2.6027412
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7316732, 2.7356372
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0376797, 2.0422637
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2547479, 2.2526457
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1638269, 3.1637597
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0252500, 2.0274520
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2620735, 2.2667992

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8652944, upper bound: 0.8715610
time: 10.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8652987, upper bound: 0.8687274
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8329086, 1.8352187
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7711763, 2.7812209
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3312640, 2.3254175
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5942540, 2.6020732
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7315578, 2.7357330
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0382605, 2.0415614
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2550921, 2.2522295
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1638613, 3.1637187
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0251074, 2.0275705
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2616754, 2.2671285

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8642016, upper bound: 0.8715619
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8642029, upper bound: 0.8687321
time: 8.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8350525, 1.8330872
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7802877, 2.7721839
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3284101, 2.3284488
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5989366, 2.5975094
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7366753, 2.7306354
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0393481, 2.0405953
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2515798, 2.2558141
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1626759, 3.1649108
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0280890, 2.0246129
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2683396, 2.2605336

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8697549, upper bound: 0.8642045
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8697567, upper bound: 0.8642034
time: 15.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 6.6375141, 9.0313435, 6.6375141, 9.0313435, -1.8350525, 1.8330750
1: -17.4851494, -13.7643299, -17.4851494, -13.7643299, -2.7805843, 2.7718129
2: -3.2759056, -0.5142198, -3.2759056, -0.5142198, -2.3273849, 2.3292971
3: -10.8677940, -7.9381065, -10.8677940, -7.9381065, -2.5994859, 2.5968413
4: -12.5387945, -9.0154028, -12.5387945, -9.0154028, -2.7365599, 2.7307312
5: -4.9653888, -2.6635816, -4.9653888, -2.6635816, -2.0399294, 2.0398927
6: -3.0826330, -0.5545907, -3.0826330, -0.5545907, -2.2519240, 2.2553978
7: -9.3434544, -5.3956985, -9.3434544, -5.3956985, -3.1627092, 3.1648698
8: -2.6018815, -0.3418674, -2.6018815, -0.3418674, -2.0279465, 2.0247312
9: -4.4801106, -1.7481177, -4.4801106, -1.7481177, -2.2679415, 2.2608631

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 453
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 4603
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 859

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8687257, upper bound: 0.8642093
time: 12.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8715560, upper bound: 0.8642080
time: 15.88 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 40.77 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8652914, upper bound: 0.8715617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8652928, upper bound: 0.8687307
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8642016, upper bound: 0.8715648
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8642029, upper bound: 0.8687342
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8652944, upper bound: 0.8715610
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8652987, upper bound: 0.8687274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8642016, upper bound: 0.8715619
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8642029, upper bound: 0.8687321
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8697549, upper bound: 0.8642045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8697567, upper bound: 0.8642034
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8687257, upper bound: 0.8642093
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 40.77
Output dim: 0, lower bound: -0.8715560, upper bound: 0.8642080
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8737239, upper bound: 0.8652849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8726967, upper bound: 0.8652872
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8652874, upper bound: 0.8726947
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8652853, upper bound: 0.8737243
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8727003, upper bound: 0.8652832
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8726964, upper bound: 0.8663674
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8727026, upper bound: 0.8652845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 40.77
Output dim: 0, lower bound: -0.8726987, upper bound: 0.8663651
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8442518711090088
rel_dist={0: [-0.8742898229747036, 0.874290832632278]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2413.94 seconds
