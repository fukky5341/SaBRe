## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.6088244805
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.6422567, 2.6422567)
1: (-7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.9443550, 2.9443550)
2: (9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.9136095, 1.9136095)
3: (-4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648)
4: (-9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.7138886, 2.7138886)
5: (-13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.5247240, 2.5247238)
6: (-16.3375587, -12.7550831, -16.3375587, -12.7550831, -3.3848834, 3.3848829)
7: (-4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302)
8: (-6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.4180570, 2.4180570)
9: (-11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.5149889, 2.5149889)

## BASE Result
execution time: IAR + LP analysis = 14.04 + 32.83 = 46.87 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3553.13 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.7896461486816406
rel_dist={2: [-0.9337359263914102, 0.9337356752465436]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.56638765335083
rel_dist={2: [-0.6118841057361983, 0.6118843662850555]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.417548656463623
rel_dist={2: [-0.3841969223018733, 0.3841966429074404]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.4919681549072266
rel_dist={2: [-0.5006622952101019, 0.500662257900629]}

## Binary Search Result
Binary search time: 208.17 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3344.96 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328289, upper bound: 1.0198362
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0198343, upper bound: 1.0328288
time: 7.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.23
Output dim: 2, lower bound: -1.0328289, upper bound: 1.0198362
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.23
Output dim: 2, lower bound: -1.0198343, upper bound: 1.0328288

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2016931, 2.2266202
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7315087, 2.7152081
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8646903, 1.8507102
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3678751, 2.3727856
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0136132, 2.0285156
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7748189, 2.7719455
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3076963, 2.3105717
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1432364, 2.1599348

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259976, upper bound: 1.0198200
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328149, upper bound: 1.0129944
time: 6.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2255511, 2.2016933
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7152076, 2.7307973
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8507099, 1.8640656
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3725667, 2.3678751
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0278482, 2.0136132
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7719455, 2.7746882
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3104401, 2.3076963
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1591976, 2.1432364

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0129947, upper bound: 1.0328151
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0198200, upper bound: 1.0259980
time: 5.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 2, lower bound: -1.0259976, upper bound: 1.0198200
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 2, lower bound: -1.0328149, upper bound: 1.0129944
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 2, lower bound: -1.0129947, upper bound: 1.0328151
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.28
Output dim: 2, lower bound: -1.0198200, upper bound: 1.0259980

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2043781, 2.2134931
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7333736, 2.7061367
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8564744, 1.8523979
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3692174, 2.3661711
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0148940, 2.0221820
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7757664, 2.7672243
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3090768, 2.3038216
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1434772, 2.1587584

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0255174, upper bound: 1.0197888
time: 11.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259680, upper bound: 1.0193446
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1885662, 2.2266202
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7224379, 2.7152081
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8646903, 1.8424945
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3612604, 2.3727856
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0072794, 2.0285156
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7700977, 2.7719455
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3009458, 2.3105717
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1420600, 2.1599348

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0323365, upper bound: 1.0129646
time: 8.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0327844, upper bound: 1.0125156
time: 7.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2282362, 2.1885662
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7170734, 2.7217255
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8424945, 1.8657534
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3739090, 2.3612604
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0291290, 2.0072794
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7728930, 2.7699666
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3118196, 2.3009458
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1594379, 2.1420600

Time for backsubstitution: 12.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0125158, upper bound: 1.0327841
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0129649, upper bound: 1.0323364
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2124243, 2.2016933
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7061367, 2.7307973
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8507099, 1.8558500
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3659520, 2.3678751
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0215144, 2.0136132
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7672243, 2.7746882
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3036885, 2.3076963
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1580212, 2.1432364

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0193427, upper bound: 1.0259680
time: 9.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0197892, upper bound: 1.0255172
time: 8.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 30.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0255174, upper bound: 1.0197888
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0259680, upper bound: 1.0193446
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0323365, upper bound: 1.0129646
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0327844, upper bound: 1.0125156
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0125158, upper bound: 1.0327841
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0129649, upper bound: 1.0323364
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0193427, upper bound: 1.0259680
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.51
Output dim: 2, lower bound: -1.0197892, upper bound: 1.0255172

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2040472, 2.2175870
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7347908, 2.7060218
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8561907, 1.8559072
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3806572, 2.3652587
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0144830, 2.0272818
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7802248, 2.7668674
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3101349, 2.3037386
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1439326, 2.1587214

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0152153, upper bound: 1.0189313
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0117041, upper bound: 1.0189375
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2043781, 2.2131622
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7332592, 2.7061367
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8564744, 1.8521137
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3683052, 2.3661711
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0148940, 2.0217705
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7754097, 2.7672243
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3089933, 2.3038216
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1434405, 2.1587584

Time for backsubstitution: 12.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0156641, upper bound: 1.0184870
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0121522, upper bound: 1.0184893
time: 7.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1882353, 2.2307141
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7238550, 2.7150927
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8644056, 1.8460062
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3727002, 2.3718734
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0068684, 2.0336156
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7745581, 2.7715893
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3020048, 2.3104882
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1425154, 2.1598978

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0220366, upper bound: 1.0121084
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0185300, upper bound: 1.0121124
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1885662, 2.2262893
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7223225, 2.7152081
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8646903, 1.8422105
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3603482, 2.3727856
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0072794, 2.0281043
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7697406, 2.7719455
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3008623, 2.3105717
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1420233, 2.1599348

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0224826, upper bound: 1.0116589
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0189755, upper bound: 1.0116635
time: 6.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2279034, 2.1926570
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7184896, 2.7216105
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8422103, 1.8692629
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3853488, 2.3603482
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0287175, 2.0123794
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7773509, 2.7696109
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3128786, 2.3008628
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1598933, 2.1420231

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0116635, upper bound: 1.0189774
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0116591, upper bound: 1.0224826
time: 35.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2282362, 2.1882353
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7169580, 2.7217255
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8424945, 1.8654695
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3729968, 2.3612604
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0291290, 2.0068681
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7725363, 2.7699666
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3117371, 2.3009458
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1594012, 2.1420600

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0121122, upper bound: 1.0185306
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0121078, upper bound: 1.0220385
time: 6.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2120914, 2.2057841
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7075529, 2.7306819
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8504257, 1.8593616
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3773913, 2.3669629
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0211024, 2.0187130
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7716846, 2.7743325
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3047476, 2.3076119
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1584761, 2.1431994

Time for backsubstitution: 12.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0184895, upper bound: 1.0121517
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0184851, upper bound: 1.0156660
time: 6.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2124243, 2.2013624
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7060213, 2.7307973
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8507099, 1.8555660
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3650393, 2.3678751
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0215144, 2.0132017
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7668676, 2.7746882
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3036070, 2.3076963
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1579840, 2.1432364

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0189357, upper bound: 1.0117039
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0189312, upper bound: 1.0152172
time: 4.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0152153, upper bound: 1.0189313
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0117041, upper bound: 1.0189375
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0156641, upper bound: 1.0184870
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0121522, upper bound: 1.0184893
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0220366, upper bound: 1.0121084
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0185300, upper bound: 1.0121124
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0224826, upper bound: 1.0116589
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0189755, upper bound: 1.0116635
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0116635, upper bound: 1.0189774
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0116591, upper bound: 1.0224826
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0121122, upper bound: 1.0185306
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0121078, upper bound: 1.0220385
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0184895, upper bound: 1.0121517
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0184851, upper bound: 1.0156660
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0189357, upper bound: 1.0117039
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0189312, upper bound: 1.0152172

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2029037, 2.2224011
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7397308, 2.7048554
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8599458, 1.8550155
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3803740, 2.3664503
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0132403, 2.0325074
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7803040, 2.7668483
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3099852, 2.3043737
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1430371, 2.1625018

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0149769, upper bound: 1.0128688
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0149769, upper bound: 1.0118880
time: 7.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2040472, 2.2164438
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7336245, 2.7060218
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8552990, 1.8559072
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3806572, 2.3649757
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0144830, 2.0260396
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7802057, 2.7668674
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3101349, 2.3035879
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1439326, 2.1578259

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114672, upper bound: 1.0128755
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114672, upper bound: 1.0118948
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2032356, 2.2179763
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7381992, 2.7049704
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8602295, 1.8512220
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3680224, 2.3673625
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0136514, 2.0269961
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7754889, 2.7672050
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3088427, 2.3044567
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1425450, 2.1625388

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0154236, upper bound: 1.0124222
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0154236, upper bound: 1.0114414
time: 6.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2043781, 2.2120187
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7320929, 2.7061367
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8555827, 1.8521137
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3683052, 2.3658881
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0148940, 2.0205283
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7753906, 2.7672243
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3089933, 2.3036709
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1434405, 2.1578629

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0119134, upper bound: 1.0124266
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0119134, upper bound: 1.0114459
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1870918, 2.2355280
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7287941, 2.7139277
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8681607, 1.8451145
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3724170, 2.3730650
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0056262, 2.0388412
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7746372, 2.7715712
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3018541, 2.3111234
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1416199, 2.1636786

Time for backsubstitution: 12.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0149962, upper bound: 1.0118711
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0159685, upper bound: 1.0118710
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1882353, 2.2295709
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7226877, 2.7150927
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8635139, 1.8460062
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3727002, 2.3715904
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0068684, 2.0323732
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7745390, 2.7715893
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3020048, 2.3103375
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1425154, 2.1590028

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114865, upper bound: 1.0118733
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124672, upper bound: 1.0118755
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1874228, 2.2311032
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7272625, 2.7140427
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8684454, 1.8413188
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3600650, 2.3739772
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0060368, 2.0333300
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7698202, 2.7719264
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3007116, 2.3112073
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1411278, 2.1637156

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0154429, upper bound: 1.0114221
time: 11.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0164138, upper bound: 1.0114224
time: 5.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1885662, 2.2251458
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7211561, 2.7152081
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8637986, 1.8422105
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3603482, 2.3725028
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0072794, 2.0268621
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7697220, 2.7719455
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3008623, 2.3104215
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1420233, 2.1590397

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0119327, upper bound: 1.0114266
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0129135, upper bound: 1.0114265
time: 6.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2267599, 2.1974678
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7234268, 2.7204418
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8459663, 1.8683701
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3850656, 2.3615386
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0274739, 2.0176053
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7774305, 2.7695920
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3127279, 2.3014975
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1589978, 2.1458035

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114267, upper bound: 1.0129135
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114267, upper bound: 1.0119327
time: 6.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2279034, 2.1915138
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7173233, 2.7216105
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8413186, 1.8692629
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3853488, 2.3600650
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0287175, 2.0111372
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7773318, 2.7696109
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3128786, 2.3007121
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1598933, 2.1411276

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114222, upper bound: 1.0164135
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114222, upper bound: 1.0154425
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2270927, 2.1930461
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7218952, 2.7205563
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8462501, 1.8645768
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3727140, 2.3624511
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0278869, 2.0120940
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7726159, 2.7699475
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3115864, 2.3015809
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1585057, 2.1458404

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118736, upper bound: 1.0124692
time: 9.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118736, upper bound: 1.0114884
time: 9.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2282362, 2.1870921
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7157917, 2.7217255
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8416028, 1.8654695
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3729968, 2.3609774
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0291290, 2.0056260
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7725172, 2.7699666
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3117371, 2.3007956
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1594012, 2.1411645

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118691, upper bound: 1.0159689
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0118691, upper bound: 1.0149966
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2109480, 2.2105947
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7124901, 2.7295136
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8541813, 1.8584690
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3771086, 2.3681533
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0198593, 2.0239389
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7717638, 2.7743139
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3045969, 2.3082471
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1575806, 2.1469798

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114460, upper bound: 1.0119137
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0124267, upper bound: 1.0119129
time: 7.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0149769, upper bound: 1.0128688
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0149769, upper bound: 1.0118880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114672, upper bound: 1.0128755
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114672, upper bound: 1.0118948
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0154236, upper bound: 1.0124222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0154236, upper bound: 1.0114414
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0119134, upper bound: 1.0124266
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0119134, upper bound: 1.0114459
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0149962, upper bound: 1.0118711
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0159685, upper bound: 1.0118710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114865, upper bound: 1.0118733
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0124672, upper bound: 1.0118755
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0154429, upper bound: 1.0114221
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0164138, upper bound: 1.0114224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0119327, upper bound: 1.0114266
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0129135, upper bound: 1.0114265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114267, upper bound: 1.0129135
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114267, upper bound: 1.0119327
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114222, upper bound: 1.0164135
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114222, upper bound: 1.0154425
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0118736, upper bound: 1.0124692
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0118736, upper bound: 1.0114884
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0118691, upper bound: 1.0159689
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0118691, upper bound: 1.0149966
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0114460, upper bound: 1.0119137
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.87
Output dim: 2, lower bound: -1.0124267, upper bound: 1.0119129
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.87
Output dim: 2, lower bound: -1.0184851, upper bound: 1.0156660
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.87
Output dim: 2, lower bound: -1.0189357, upper bound: 1.0117039
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.87
Output dim: 2, lower bound: -1.0189312, upper bound: 1.0152172
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8640661239624023
rel_dist={2: [-1.032862430954955, 1.0328621621306109]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218423, upper bound: 0.7143239
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7143245, upper bound: 0.7218444
time: 5.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.07
Output dim: 2, lower bound: -0.7218423, upper bound: 0.7143239
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.07
Output dim: 2, lower bound: -0.7143245, upper bound: 0.7218444

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8841081, 1.8983521
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4262118, 2.4168973
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6354403, 1.6274519
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9991493, 1.9948106
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0629687, 2.0657749
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7154875, 1.7240033
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4074707, 2.4058290
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5398674, 2.5403514
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0963297, 2.0979729
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8287756, 1.8383174

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7173626, upper bound: 0.7143222
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218379, upper bound: 0.7098469
time: 4.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8983521, 1.8841081
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4168973, 2.4262123
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6274514, 1.6354403
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9948111, 1.9991493
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0657744, 2.0629683
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7240033, 1.7154877
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4058285, 2.4074707
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5403509, 2.5398674
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0979729, 2.0963297
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8383176, 1.8287756

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7098448, upper bound: 0.7218380
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7143201, upper bound: 0.7173647
time: 5.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.83
Output dim: 2, lower bound: -0.7173626, upper bound: 0.7143222
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.83
Output dim: 2, lower bound: -0.7218379, upper bound: 0.7098469
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.83
Output dim: 2, lower bound: -0.7098448, upper bound: 0.7218380
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.83
Output dim: 2, lower bound: -0.7143201, upper bound: 0.7173647

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8800163, 1.8852251
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4233904, 2.4078259
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6272244, 1.6248951
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9981179, 1.9915180
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0609012, 2.0591602
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7135053, 1.7176697
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4059887, 2.4011075
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5385303, 2.5360694
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0942254, 2.0912228
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8284090, 1.8371410

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7168749, upper bound: 0.7142905
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7173329, upper bound: 0.7138316
time: 6.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8709812, 1.8942606
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4171410, 2.4140754
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6328835, 1.6192360
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9958568, 1.9937797
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0563540, 2.0637069
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7091541, 1.7220211
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4027495, 2.4043469
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5355854, 2.5390139
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0895791, 2.0958686
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8275993, 1.8379509

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7213501, upper bound: 0.7098150
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218079, upper bound: 0.7093540
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8942604, 1.8709810
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4140749, 2.4171410
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6192360, 1.6328835
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9937797, 1.9958563
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0637069, 2.0563540
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7220211, 1.7091541
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4043469, 2.4027495
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5390139, 2.5355854
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0958695, 2.0895791
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8379509, 1.8275993

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7093543, upper bound: 0.7218074
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7098152, upper bound: 0.7213501
time: 9.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8852253, 1.8800166
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4078255, 2.4233904
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6248951, 1.6272244
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9915175, 1.9981179
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0591598, 2.0609007
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7176700, 1.7135053
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4011078, 2.4059889
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5360699, 2.5385303
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0912232, 2.0942254
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8371413, 1.8284090

Time for backsubstitution: 12.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7138296, upper bound: 0.7173350
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7142903, upper bound: 0.7168745
time: 8.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7168749, upper bound: 0.7142905
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7173329, upper bound: 0.7138316
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7213501, upper bound: 0.7098150
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7218079, upper bound: 0.7093540
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7093543, upper bound: 0.7218074
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7098152, upper bound: 0.7213501
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7138296, upper bound: 0.7173350
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.24
Output dim: 2, lower bound: -0.7142903, upper bound: 0.7168745

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8796854, 1.8874226
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4241505, 2.4077110
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6269407, 1.6267786
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9985633, 1.9914503
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0670466, 2.0582476
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7130938, 1.7204077
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4083834, 2.4007506
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5417018, 2.5355897
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0947943, 2.0911398
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8286536, 1.8371041

Time for backsubstitution: 12.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7118743, upper bound: 0.7133625
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7084458, upper bound: 0.7133669
time: 7.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8800163, 1.8848941
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4232750, 2.4078259
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6272244, 1.6246109
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9980502, 1.9915180
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0599885, 2.0591602
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7135053, 1.7172585
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4056320, 2.4011075
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5380511, 2.5360694
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0941420, 2.0912228
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8283722, 1.8371410

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7123335, upper bound: 0.7129015
time: 9.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089063, upper bound: 0.7129055
time: 9.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8706503, 1.8964581
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4179010, 2.4139605
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6325998, 1.6211209
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9963012, 1.9937119
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0624995, 2.0627947
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7087426, 1.7247608
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4051456, 2.4039903
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5387597, 2.5385342
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0901489, 2.0957856
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8278439, 1.8379140

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7163496, upper bound: 0.7088877
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129211, upper bound: 0.7088919
time: 9.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8709812, 1.8939297
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4170256, 2.4140754
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6328835, 1.6189518
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9957881, 1.9937797
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0554414, 2.0637069
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7091541, 1.7216096
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4023924, 2.4043469
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5351062, 2.5390139
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0894957, 2.0958686
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8275626, 1.8379509

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7168085, upper bound: 0.7084265
time: 18.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133814, upper bound: 0.7084310
time: 11.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8939295, 1.8731768
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4148359, 2.4170260
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6189518, 1.6347666
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9942241, 1.9957886
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0698524, 2.0554419
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7216096, 1.7118921
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4067411, 2.4023926
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5421863, 2.5351057
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0964384, 2.0894961
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8381941, 1.8275623

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7084311, upper bound: 0.7133809
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7084267, upper bound: 0.7168086
time: 8.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8942604, 1.8706501
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4139605, 2.4171410
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6192360, 1.6325998
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9937119, 1.9958563
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0627947, 2.0563540
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7220211, 1.7087426
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4039903, 2.4027495
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5385346, 2.5355854
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0957861, 2.0895791
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8379138, 1.8275993

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7088921, upper bound: 0.7129208
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7088876, upper bound: 0.7163517
time: 8.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8848944, 1.8822124
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4085855, 2.4232755
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6246109, 1.6291084
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9919629, 1.9980502
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0653057, 2.0599885
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7172585, 1.7162452
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4035034, 2.4056320
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5392432, 2.5380507
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0917921, 2.0941424
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8373845, 1.8283720

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129064, upper bound: 0.7089058
time: 20.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129020, upper bound: 0.7123333
time: 6.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8852253, 1.8796856
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4077110, 2.4233904
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6248951, 1.6269407
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9914498, 1.9981179
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0582480, 2.0609007
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7176700, 1.7130940
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4007506, 2.4059889
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5355897, 2.5385303
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0911398, 2.0942254
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8371041, 1.8284090

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133672, upper bound: 0.7084454
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133627, upper bound: 0.7118764
time: 6.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7118743, upper bound: 0.7133625
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7084458, upper bound: 0.7133669
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7123335, upper bound: 0.7129015
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7089063, upper bound: 0.7129055
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7163496, upper bound: 0.7088877
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7129211, upper bound: 0.7088919
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7168085, upper bound: 0.7084265
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7133814, upper bound: 0.7084310
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7084311, upper bound: 0.7133809
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7084267, upper bound: 0.7168086
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7088921, upper bound: 0.7129208
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7088876, upper bound: 0.7163517
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7129064, upper bound: 0.7089058
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7129020, upper bound: 0.7123333
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7133672, upper bound: 0.7084454
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.03
Output dim: 2, lower bound: -0.7133627, upper bound: 0.7118764

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8785419, 1.8896835
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4264727, 2.4065447
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6287041, 1.6258869
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9998178, 1.9908190
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0667639, 2.0588074
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7118516, 1.7228613
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4084206, 2.4007316
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5414681, 2.5360541
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0946436, 2.0914383
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8277581, 1.8388805

Time for backsubstitution: 12.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7115840, upper bound: 0.7096415
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7115839, upper bound: 0.7086126
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8796854, 1.8862793
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4229841, 2.4077110
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6260490, 1.6267786
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9979324, 1.9914503
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0670466, 2.0579648
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7130938, 1.7191653
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4083643, 2.4007506
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5417018, 2.5353560
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0947943, 2.0909891
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8286536, 1.8362088

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081546, upper bound: 0.7096481
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081545, upper bound: 0.7086175
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8788738, 1.8871551
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4255981, 2.4066596
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6289883, 1.6237192
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9993048, 1.9908867
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0597057, 2.0597196
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7122626, 1.7197123
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4056692, 2.4010882
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5378175, 2.5365338
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0939922, 2.0915213
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8274767, 1.8389175

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7120433, upper bound: 0.7091826
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7120432, upper bound: 0.7081538
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8800163, 1.8837507
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4221087, 2.4078259
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6263328, 1.6246109
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9974194, 1.9915180
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0599885, 2.0588770
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7135053, 1.7160163
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4056129, 2.4011075
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5380511, 2.5358357
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0941420, 2.0910721
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8283722, 1.8362458

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086153, upper bound: 0.7091851
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086152, upper bound: 0.7081582
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8695068, 1.8987191
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4202232, 2.4127941
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6343632, 1.6202292
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9975567, 1.9930806
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0622168, 2.0633540
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7075005, 1.7272143
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4051828, 2.4039712
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5385261, 2.5389986
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0899992, 2.0960846
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8269484, 1.8396904

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7116002, upper bound: 0.7085963
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7126284, upper bound: 0.7085964
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8706503, 1.8953149
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4167347, 2.4139605
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6317081, 1.6211209
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9956703, 1.9937119
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0624995, 2.0625119
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7087426, 1.7235184
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4051266, 2.4039903
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5387597, 2.5383010
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0901489, 2.0956354
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8278439, 1.8370185

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081708, upper bound: 0.7086007
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7091997, upper bound: 0.7086006
time: 6.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8698378, 1.8961904
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4193487, 2.4129090
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6346469, 1.6180601
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9970436, 1.9931488
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0551586, 2.0642667
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7079115, 1.7240634
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4024296, 2.4043279
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5348725, 2.5394783
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0893459, 2.0961676
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8266671, 1.8397274

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7120594, upper bound: 0.7081351
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7130875, upper bound: 0.7081349
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8709812, 1.8927863
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4158592, 2.4140754
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6319919, 1.6189518
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9951572, 1.9937797
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0554414, 2.0634241
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7091541, 1.7203674
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4023738, 2.4043469
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5351062, 2.5387807
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0894957, 2.0957184
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8275626, 1.8370554

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086315, upper bound: 0.7081394
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7096602, upper bound: 0.7081420
time: 7.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8927860, 1.8754361
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4171562, 2.4158597
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6207161, 1.6338749
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9954796, 1.9951572
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0695696, 2.0560007
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7203674, 1.7143459
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4067783, 2.4023736
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5419526, 2.5355701
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0962877, 2.0897946
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8372986, 1.8293386

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081399, upper bound: 0.7096604
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081398, upper bound: 0.7086312
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8939295, 1.8720336
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4136686, 2.4170260
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6180601, 1.6347666
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9935932, 1.9957886
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0698524, 2.0551586
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7216096, 1.7106497
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4067221, 2.4023926
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5421863, 2.5348725
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0964384, 2.0893459
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8381941, 1.8266668

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081355, upper bound: 0.7130876
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7081354, upper bound: 0.7120588
time: 9.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8931170, 1.8729093
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4162807, 2.4159746
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6209998, 1.6317081
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9949675, 1.9952250
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0625114, 2.0569129
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7207785, 1.7111967
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4040275, 2.4027302
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5383010, 2.5360498
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0956354, 2.0898781
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8370183, 1.8293755

Time for backsubstitution: 12.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086010, upper bound: 0.7091992
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086009, upper bound: 0.7081701
time: 8.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8942604, 1.8695068
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4127941, 2.4171410
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6183443, 1.6325998
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9930811, 1.9958563
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0627947, 2.0560708
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7220211, 1.7075005
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4039712, 2.4027495
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5385346, 2.5353522
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0957861, 2.0894289
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8379138, 1.8267038

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7085966, upper bound: 0.7126304
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7085965, upper bound: 0.7115999
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8837509, 1.8844714
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4109068, 2.4221091
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6263752, 1.6282167
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9932175, 1.9974189
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0650225, 2.0605478
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7160163, 1.7186990
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4035406, 2.4056129
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5390096, 2.5385146
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0916414, 2.0944409
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8364890, 1.8301482

Time for backsubstitution: 12.64 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.6408071517944336
rel_dist={2: [-0.7218570837425542, 0.7218564741351408]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118731, upper bound: 0.6062239
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6062217, upper bound: 0.6118729
time: 7.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.49
Output dim: 2, lower bound: -0.6118731, upper bound: 0.6062239
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.49
Output dim: 2, lower bound: -0.6062217, upper bound: 0.6118729

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7782464, 1.7889295
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3244462, 2.3174601
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5590239, 1.5530319
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9176044, 1.9143505
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9613328, 1.9634376
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6161127, 1.6224992
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2850213, 2.2837901
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4624062, 2.4627686
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0258746, 2.0271068
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7239554, 1.7311115

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6085122, upper bound: 0.6062182
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118698, upper bound: 0.6028630
time: 5.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7889295, 1.7782464
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3174605, 2.3244467
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5530319, 1.5590239
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9143505, 1.9176044
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9634376, 1.9613333
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6224990, 1.6161127
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2837896, 2.2850213
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4627686, 2.4624057
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0271072, 2.0258741
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7311118, 1.7239552

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6028608, upper bound: 0.6118720
time: 15.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6062185, upper bound: 0.6085143
time: 5.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 33.71 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 33.71
Output dim: 2, lower bound: -0.6085122, upper bound: 0.6062182
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 33.71
Output dim: 2, lower bound: -0.6118698, upper bound: 0.6028630
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 33.71
Output dim: 2, lower bound: -0.6028608, upper bound: 0.6118720
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 33.71
Output dim: 2, lower bound: -0.6062185, upper bound: 0.6085143

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7651196, 1.7825789
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3153744, 2.3130760
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5550523, 1.5448165
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9143119, 1.9127541
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9547186, 1.9602332
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6097789, 1.6194291
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2803001, 2.2814982
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4581242, 2.4606953
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0191240, 2.0238414
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7227790, 1.7305427

Time for backsubstitution: 12.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6113806, upper bound: 0.6028341
time: 16.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6118408, upper bound: 0.6023707
time: 8.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7825785, 1.7651193
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3130760, 2.3153753
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5448165, 1.5550523
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9127545, 1.9143119
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9602332, 1.9547186
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6194291, 1.6097789
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2814980, 2.2803001
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4606953, 2.4581242
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0238409, 2.0191240
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7305429, 1.7227788

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6023712, upper bound: 0.6118404
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.6028320, upper bound: 0.6113808
time: 9.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.33 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 2, lower bound: -0.6113806, upper bound: 0.6028341
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 2, lower bound: -0.6118408, upper bound: 0.6023707
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 2, lower bound: -0.6023712, upper bound: 0.6118404
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.33
Output dim: 2, lower bound: -0.6028320, upper bound: 0.6113808

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7647886, 1.7841444
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3159170, 2.3129611
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5547681, 1.5461593
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9146285, 1.9126863
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9590998, 1.9593210
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6093674, 1.6213810
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2820082, 2.2811415
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4603848, 2.4602156
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0195303, 2.0237584
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7229531, 1.7305057

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6082566, upper bound: 0.6019046
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6048156, upper bound: 0.6019099
time: 9.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7651196, 1.7822480
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3152599, 2.3130760
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5550523, 1.5445323
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9142442, 1.9127541
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9538064, 1.9602332
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6097789, 1.6190178
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2799435, 2.2814982
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4576440, 2.4606953
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0190411, 2.0238414
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7227423, 1.7305427

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6087152, upper bound: 0.6014438
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6052758, upper bound: 0.6014467
time: 7.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7822475, 1.7666833
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3136177, 2.3152604
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5445323, 1.5563931
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9130702, 1.9142442
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9646144, 1.9538059
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6190176, 1.6117296
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2832050, 2.2799432
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4629540, 2.4576445
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0242472, 2.0190411
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7307160, 1.7227418

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6014469, upper bound: 0.6052755
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6014436, upper bound: 0.6087148
time: 6.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.7825785, 1.7647884
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.3129606, 2.3153753
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.5448165, 1.5547681
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9126868, 1.9143119
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -1.9593210, 1.9547186
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6194291, 1.6093676
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.2811413, 2.2803001
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.4602160, 2.4581242
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0237584, 2.0191240
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.7305062, 1.7227788

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019077, upper bound: 0.6048155
time: 12.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6019044, upper bound: 0.6082565
time: 5.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.41 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6082566, upper bound: 0.6019046
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6048156, upper bound: 0.6019099
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6087152, upper bound: 0.6014438
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6052758, upper bound: 0.6014467
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6014469, upper bound: 0.6052755
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6014436, upper bound: 0.6087148
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6019077, upper bound: 0.6048155
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 31.41
Output dim: 2, lower bound: -0.6019044, upper bound: 0.6082565
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.56638765335083
rel_dist={2: [-0.6118841057361983, 0.6118843662850555]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1921.26 seconds
