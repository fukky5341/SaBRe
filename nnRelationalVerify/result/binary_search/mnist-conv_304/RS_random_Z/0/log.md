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
execution time: IAR + LP analysis = 16.63 + 33.27 = 49.91 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3550.09 seconds, max iter: 100)

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
Binary search time: 219.04 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3331.06 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6218

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0328623, upper bound: 1.0324757
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0324758, upper bound: 1.0328618
time: 8.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.73
Output dim: 2, lower bound: -1.0328623, upper bound: 1.0324757
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.73
Output dim: 2, lower bound: -1.0324758, upper bound: 1.0328618

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2235212, 2.2262735
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7149820, 2.7084637
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8632112, 1.8627009
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3752770, 2.3761454
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0266132, 2.0240254
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7595553, 2.7639680
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3091831, 2.3100171
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1472797, 2.1507492

Time for backsubstitution: 15.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0264058, upper bound: 1.0264062
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0267929, upper bound: 1.0260191
time: 6.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2262735, 2.2235212
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7084637, 2.7149820
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8627009, 1.8632112
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3761454, 2.3752768
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0240254, 2.0266132
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7639680, 2.7595553
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3100176, 2.3091831
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1507492, 2.1472797

Time for backsubstitution: 13.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0319968, upper bound: 1.0328316
time: 8.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0324454, upper bound: 1.0323832
time: 8.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 30.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.80
Output dim: 2, lower bound: -1.0264058, upper bound: 1.0264062
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.80
Output dim: 2, lower bound: -1.0267929, upper bound: 1.0260191
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.80
Output dim: 2, lower bound: -1.0319968, upper bound: 1.0328316
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.80
Output dim: 2, lower bound: -1.0324454, upper bound: 1.0323832

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2266202, 2.2259126
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7197046, 2.7079153
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8629699, 1.8647709
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3801432, 2.3755794
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0309348, 2.0235188
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7613945, 2.7637513
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3091650, 2.3101735
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1468344, 2.1545601

Time for backsubstitution: 15.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0236178, upper bound: 1.0264024
time: 10.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0264026, upper bound: 1.0236310
time: 8.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2231603, 2.2262735
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7144337, 2.7084637
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8632112, 1.8624597
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3747106, 2.3761454
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0261068, 2.0240254
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7593389, 2.7639680
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3091831, 2.3099980
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1472797, 2.1503041

Time for backsubstitution: 15.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0240157, upper bound: 1.0260180
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0267891, upper bound: 1.0232314
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2259412, 2.2276106
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7098813, 2.7148676
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8624167, 1.8667228
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3875856, 2.3743651
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0236135, 2.0317128
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7684288, 2.7591996
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3110743, 2.3090992
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1512051, 2.1472430

Time for backsubstitution: 15.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0247560, upper bound: 1.0328185
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0319834, upper bound: 1.0255929
time: 8.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2262735, 2.2231889
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7083488, 2.7149820
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8627009, 1.8629272
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3752337, 2.3752768
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0240254, 2.0262015
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7636118, 2.7595553
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3099337, 2.3091831
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1507125, 2.1472797

Time for backsubstitution: 15.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0296562, upper bound: 1.0323803
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0324423, upper bound: 1.0295938
time: 5.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 29.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0236178, upper bound: 1.0264024
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0264026, upper bound: 1.0236310
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0240157, upper bound: 1.0260180
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0267891, upper bound: 1.0232314
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0247560, upper bound: 1.0328185
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0319834, upper bound: 1.0255929
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0296562, upper bound: 1.0323803
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.40
Output dim: 2, lower bound: -1.0324423, upper bound: 1.0295938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2196069, 2.2154744
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7178411, 2.7016430
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8544936, 1.8594925
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3767633, 2.3693569
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0298729, 2.0214605
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7573404, 2.7562759
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3023486, 2.3005481
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1380701, 2.1483564

Time for backsubstitution: 15.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0235843, upper bound: 1.0133907
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0105837, upper bound: 1.0263686
time: 7.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2161822, 2.2189004
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7134314, 2.7060528
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8576941, 1.8562951
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3739209, 2.3722017
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0288768, 2.0224569
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7539186, 2.7596996
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2995391, 2.3033576
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1406326, 2.1457958

Time for backsubstitution: 15.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0191617, upper bound: 1.0236161
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0263892, upper bound: 1.0163878
time: 11.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2161479, 2.2158346
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7125702, 2.7021923
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8547359, 1.8571837
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3713331, 2.3699226
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0250444, 2.0219674
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7552867, 2.7564929
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3023677, 2.3003726
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1385140, 2.1441021

Time for backsubstitution: 12.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0235349, upper bound: 1.0259873
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0239860, upper bound: 1.0255395
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2127223, 2.2192607
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7081614, 2.7066011
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8579364, 1.8539836
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3684883, 2.3727674
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0240483, 2.0229635
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7518630, 2.7599168
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2995582, 2.3031821
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1410766, 2.1415396

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0263122, upper bound: 1.0232036
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0267583, upper bound: 1.0227523
time: 5.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2251701, 2.2273307
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7090421, 2.7145662
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8497825, 1.8621843
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3867297, 2.3740575
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0166464, 2.0123234
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7664957, 2.7538249
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3048334, 2.2917523
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1456652, 2.1452482

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0179229, upper bound: 1.0328061
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0247420, upper bound: 1.0259897
time: 7.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2256613, 2.2268395
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7095799, 2.7140284
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8578782, 1.8540885
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3872781, 2.3735092
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0042243, 2.0247455
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7630548, 2.7572660
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2937279, 2.3028579
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1492100, 2.1417034

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0319505, upper bound: 1.0125628
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0189574, upper bound: 1.0255578
time: 7.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2192607, 2.2127504
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7064867, 2.7087107
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8542256, 1.8576515
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3718557, 2.3690541
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0229635, 2.0241423
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7595606, 2.7520804
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3031173, 2.2995572
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1419463, 2.1410766

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0232018, upper bound: 1.0263140
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0235995, upper bound: 1.0259275
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2158346, 2.2161763
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7020769, 2.7131195
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8574252, 1.8544517
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3690109, 2.3718987
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0219674, 2.0251386
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7561369, 2.7555041
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3003078, 2.3023667
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1445088, 2.1385140

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0259855, upper bound: 1.0235347
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0263718, upper bound: 1.0231388
time: 6.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0235843, upper bound: 1.0133907
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0105837, upper bound: 1.0263686
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0191617, upper bound: 1.0236161
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0263892, upper bound: 1.0163878
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0235349, upper bound: 1.0259873
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0239860, upper bound: 1.0255395
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0263122, upper bound: 1.0232036
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0267583, upper bound: 1.0227523
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0179229, upper bound: 1.0328061
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0247420, upper bound: 1.0259897
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0319505, upper bound: 1.0125628
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0189574, upper bound: 1.0255578
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0232018, upper bound: 1.0263140
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0235995, upper bound: 1.0259275
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0259855, upper bound: 1.0235347
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -1.0263718, upper bound: 1.0231388

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.1957493, 2.2165437
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7185535, 2.6860542
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8551178, 1.8461361
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3720727, 2.3695769
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0156369, 2.0221272
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7574711, 2.7535334
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2996049, 2.3006802
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1221085, 2.1490932

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0231040, upper bound: 1.0133597
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0235550, upper bound: 1.0129133
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2196069, 2.1916170
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7022524, 2.7016430
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8411379, 1.8594925
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3767633, 2.3646662
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0298729, 2.0072248
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7545981, 2.7562759
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3023486, 2.2978044
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1380701, 2.1323948

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0097314, upper bound: 1.0125768
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0097267, upper bound: 1.0160744
time: 8.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2154117, 2.2186210
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7125931, 2.7057514
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8450608, 1.8517575
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3730650, 2.3718941
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0219097, 2.0030675
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7519851, 2.7543249
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2932978, 2.2860112
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1350918, 2.1437998

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0186848, upper bound: 1.0235864
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0191311, upper bound: 1.0231350
time: 8.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2159028, 2.2181299
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7131300, 2.7052135
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8531566, 1.8436615
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3736134, 2.3713458
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0094872, 2.0154896
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7485442, 2.7577660
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2821932, 2.2971168
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1386366, 2.1402550

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0257708, upper bound: 1.0163687
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0257896, upper bound: 1.0153563
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2158155, 2.2199242
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7139874, 2.7020774
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8544517, 1.8606949
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3827682, 2.3690109
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0246320, 2.0270658
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7597489, 2.7561367
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3034244, 2.3002892
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1389689, 2.1440649

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0132391, upper bound: 1.0139553
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0114961, upper bound: 1.0156936
time: 6.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2161479, 2.2155027
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7124557, 2.7021923
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8547359, 1.8568993
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3704209, 2.3699226
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0250444, 2.0215545
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7549314, 2.7564929
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3022828, 2.3003726
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1384768, 2.1441021

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0167452, upper bound: 1.0255257
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0239726, upper bound: 1.0183001
time: 6.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2123899, 2.2233522
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7095785, 2.7064867
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8576512, 1.8574951
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3799281, 2.3718555
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0236359, 2.0280621
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7563248, 2.7595603
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3006148, 2.3030987
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1415315, 2.1415024

Time for backsubstitution: 12.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0253073, upper bound: 1.0225871
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0262921, upper bound: 1.0225676
time: 7.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2127223, 2.2189286
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7080469, 2.7066011
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8579364, 1.8536994
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3675761, 2.3727674
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0240483, 2.0225508
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7515078, 2.7599168
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2994733, 2.3031821
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1410394, 2.1415396

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0195175, upper bound: 1.0227389
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0267449, upper bound: 1.0155115
time: 7.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2278552, 2.2142034
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7109065, 2.7054944
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8415675, 1.8638701
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3880720, 2.3674428
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0179272, 2.0059891
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7674403, 2.7491031
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3062139, 2.2850022
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1459060, 2.1440713

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0076262, upper bound: 1.0207632
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0058813, upper bound: 1.0225056
time: 8.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2120428, 2.2273307
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.6999698, 2.7145662
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8497825, 1.8539691
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3801150, 2.3740575
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0103121, 2.0123234
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7617736, 2.7538249
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2980828, 2.2917523
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1444888, 2.1452482

Time for backsubstitution: 12.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0177014, upper bound: 1.0257450
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0186682, upper bound: 1.0257429
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2018056, 2.2279139
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7102919, 2.6984396
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8585029, 1.8407331
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3825850, 2.3737268
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.9899898, 2.0254135
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7631860, 2.7545238
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2909832, 2.3029895
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1332483, 2.1424406

Time for backsubstitution: 12.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0254959, upper bound: 1.0065077
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0258831, upper bound: 1.0061146
time: 7.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2256613, 2.2029839
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.6939907, 2.7140284
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8445230, 1.8540885
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3872781, 2.3688161
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0042243, 2.0105112
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7603126, 2.7572660
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.2937279, 2.3001132
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1492100, 2.1257422

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0161627, upper bound: 1.0255543
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0189548, upper bound: 1.0227685
time: 15.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -2.2223592, 2.2123899
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.7112074, 2.7081614
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.8539834, 1.8597186
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.1355648, 2.1355648
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.3767200, 2.3684883
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -2.0272846, 2.0236356
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.7613974, 2.7518635
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.6866302, 2.6866302
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.3030982, 2.2997146
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -2.1415029, 2.1448867

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0225677, upper bound: 1.0262922
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0225853, upper bound: 1.0253071
time: 5.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0231040, upper bound: 1.0133597
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0235550, upper bound: 1.0129133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0097314, upper bound: 1.0125768
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0097267, upper bound: 1.0160744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0186848, upper bound: 1.0235864
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0191311, upper bound: 1.0231350
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0257708, upper bound: 1.0163687
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0257896, upper bound: 1.0153563
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0132391, upper bound: 1.0139553
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0114961, upper bound: 1.0156936
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0167452, upper bound: 1.0255257
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0239726, upper bound: 1.0183001
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0253073, upper bound: 1.0225871
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0262921, upper bound: 1.0225676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0195175, upper bound: 1.0227389
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0267449, upper bound: 1.0155115
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0076262, upper bound: 1.0207632
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0058813, upper bound: 1.0225056
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0177014, upper bound: 1.0257450
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0186682, upper bound: 1.0257429
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0254959, upper bound: 1.0065077
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0258831, upper bound: 1.0061146
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0161627, upper bound: 1.0255543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0189548, upper bound: 1.0227685
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0225677, upper bound: 1.0262922
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.87
Output dim: 2, lower bound: -1.0225853, upper bound: 1.0253071
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.87
Output dim: 2, lower bound: -1.0235995, upper bound: 1.0259275
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.87
Output dim: 2, lower bound: -1.0259855, upper bound: 1.0235347
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.87
Output dim: 2, lower bound: -1.0263718, upper bound: 1.0231388
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8640661239624023
rel_dist={2: [-1.032862430954955, 1.0328621621306109]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5762

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7173774, upper bound: 0.7218529
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218527, upper bound: 0.7173795
time: 5.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.34
Output dim: 2, lower bound: -0.7173774, upper bound: 0.7218529
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.34
Output dim: 2, lower bound: -0.7218527, upper bound: 0.7173795

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.9038744, 1.8948388
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4296637, 2.4234142
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6325917, 1.6382508
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -2.0010300, 1.9987683
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0655923, 2.0610456
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7277403, 1.7233889
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4070897, 2.4038498
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5393410, 2.5363965
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0969691, 2.0923228
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8443696, 1.8435597

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7173626, upper bound: 0.7143222
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7098448, upper bound: 0.7218380
time: 6.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8948383, 1.9038744
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4234142, 2.4296637
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6382508, 1.6325917
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9987688, 2.0010300
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0610452, 2.0655923
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7233887, 1.7277403
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4038501, 2.4070892
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5363960, 2.5393410
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0923228, 2.0969691
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8435600, 1.8443696

Time for backsubstitution: 14.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7218379, upper bound: 0.7098469
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7143201, upper bound: 0.7173647
time: 5.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -0.7173626, upper bound: 0.7143222
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -0.7098448, upper bound: 0.7218380
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -0.7218379, upper bound: 0.7098469
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.43
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

Time for backsubstitution: 13.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7170703, upper bound: 0.7106010
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7170702, upper bound: 0.7095697
time: 9.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089218, upper bound: 0.7134111
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089171, upper bound: 0.7168398
time: 7.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7168397, upper bound: 0.7089170
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7134111, upper bound: 0.7089214
time: 7.68 seconds

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

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133971, upper bound: 0.7089354
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133925, upper bound: 0.7123636
time: 7.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7170703, upper bound: 0.7106010
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7170702, upper bound: 0.7095697
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7089218, upper bound: 0.7134111
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7089171, upper bound: 0.7168398
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7168397, upper bound: 0.7089170
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7134111, upper bound: 0.7089214
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7133971, upper bound: 0.7089354
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.16
Output dim: 2, lower bound: -0.7133925, upper bound: 0.7123636

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8816328, 1.8848650
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4258604, 2.4072847
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6269836, 1.6259747
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9979162, 1.9924226
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0634384, 2.0585938
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7157693, 1.7171626
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4069571, 2.4008904
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5383701, 2.5367918
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0942073, 2.0913043
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8279636, 1.8391278

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7170702, upper bound: 0.7103730
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7168447, upper bound: 0.7105984
time: 7.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8796568, 1.8852251
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4228487, 2.4078259
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6272244, 1.6246538
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9981179, 1.9913158
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0603347, 2.0591602
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7129984, 1.7176697
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4057722, 2.4011075
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5385303, 2.5359097
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0942254, 2.0912042
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8284090, 1.8366957

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129334, upper bound: 0.7095635
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7170618, upper bound: 0.7054351
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8931170, 1.8732402
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4163971, 2.4159746
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6209998, 1.6319919
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9950352, 1.9952250
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0634241, 2.0569129
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7207785, 1.7116075
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4043837, 2.4027302
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5387802, 2.5360498
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0957189, 2.0898781
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8370554, 1.8293755

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086306, upper bound: 0.7096901
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086305, upper bound: 0.7086604
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8942604, 1.8698380
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4129086, 2.4171410
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6183443, 1.6328835
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9931488, 1.9958563
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0637069, 2.0560708
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7220211, 1.7079115
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4043274, 2.4027495
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5390139, 2.5353522
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0958695, 2.0894289
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8379509, 1.8267038

Time for backsubstitution: 13.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7072606, upper bound: 0.7168384
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7089158, upper bound: 0.7151848
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8698378, 1.8965216
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4194641, 2.4129090
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6346469, 1.6183443
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9971113, 1.9931488
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0560708, 2.0642667
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7079115, 1.7244742
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4027863, 2.4043279
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5353518, 2.5394783
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0894284, 2.0961676
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8267038, 1.8397274

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7127029, upper bound: 0.7089083
time: 14.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7168313, upper bound: 0.7047800
time: 14.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8709812, 1.8931174
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4159737, 2.4140754
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6319919, 1.6192360
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9952250, 1.9937797
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0563540, 2.0634241
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7091541, 1.7207785
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4027300, 2.4043469
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5355854, 2.5387807
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0895791, 2.0957184
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8275993, 1.8370554

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129211, upper bound: 0.7088919
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133814, upper bound: 0.7084310
time: 11.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8840818, 1.8822758
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4101467, 2.4222240
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6266589, 1.6263328
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9927731, 1.9974871
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0588770, 2.0614600
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7164273, 1.7159586
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4011440, 2.4059696
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5358362, 2.5389943
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0910726, 2.0945239
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8362458, 1.8301852

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129064, upper bound: 0.7089058
time: 18.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133672, upper bound: 0.7084454
time: 10.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8852253, 1.8788736
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4066601, 2.4233904
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6240034, 1.6272244
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9908867, 1.9981179
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0591598, 2.0606179
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7176700, 1.7122626
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4010882, 2.4059889
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5360699, 2.5382967
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0912232, 2.0940752
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8371413, 1.8275135

Time for backsubstitution: 14.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129020, upper bound: 0.7123333
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7133627, upper bound: 0.7118764
time: 5.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7170702, upper bound: 0.7103730
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7168447, upper bound: 0.7105984
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7129334, upper bound: 0.7095635
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7170618, upper bound: 0.7054351
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7086306, upper bound: 0.7096901
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7086305, upper bound: 0.7086604
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7072606, upper bound: 0.7168384
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7089158, upper bound: 0.7151848
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7127029, upper bound: 0.7089083
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7168313, upper bound: 0.7047800
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7129211, upper bound: 0.7088919
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7133814, upper bound: 0.7084310
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7129064, upper bound: 0.7089058
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7133672, upper bound: 0.7084454
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7129020, upper bound: 0.7123333
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 2, lower bound: -0.7133627, upper bound: 0.7118764

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8796031, 1.8844075
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4072523, 2.3849521
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6259103, 1.6246099
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9842639, 1.9760394
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0661488, 2.0618005
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7134261, 1.7133405
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.3918238, 2.3882787
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5233536, 2.5243282
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0929499, 2.0905228
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8160462, 1.8291934

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7154113, upper bound: 0.7103732
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7170687, upper bound: 0.7087173
time: 5.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8811758, 1.8828349
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4035273, 2.3886766
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6256185, 1.6249018
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9815335, 1.9787707
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0666456, 2.0613046
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7119470, 1.7148194
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.3943453, 2.3857574
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5259075, 2.5217752
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0934258, 2.0900464
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8180289, 1.8272107

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7151858, upper bound: 0.7105961
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7168432, upper bound: 0.7089421
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8788862, 1.8847356
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4220085, 2.4072928
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6145911, 1.6166461
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9960027, 1.9899755
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0594783, 2.0586171
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7007074, 1.6982799
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4023638, 2.3957326
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5150719, 2.5210357
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0832253, 2.0738578
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8228681, 1.8331821

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7124457, upper bound: 0.7095341
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129037, upper bound: 0.7090729
time: 6.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8791676, 1.8844550
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4223156, 2.4069858
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6192169, 1.6120198
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9967780, 1.9892001
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0597916, 2.0583038
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6936088, 1.7053781
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4003973, 2.3976989
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5236568, 2.5124502
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0768795, 2.0802040
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8248937, 1.8311565

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7120656, upper bound: 0.7045074
time: 9.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086362, upper bound: 0.7045121
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8947344, 1.8728805
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4188671, 2.4154339
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6207585, 1.6330719
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9948335, 1.9961295
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0659618, 2.0563469
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7230420, 1.7111003
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4053521, 2.4025133
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5386200, 2.5367718
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0957007, 2.0899601
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8366106, 1.8313625

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7069773, upper bound: 0.7096900
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086291, upper bound: 0.7080339
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8927574, 1.8732402
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4158564, 2.4159746
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6209998, 1.6317506
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9950352, 1.9950233
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0628576, 2.0569129
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7202711, 1.7116075
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.4041672, 2.4027302
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5387802, 2.5358896
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0957189, 2.0898600
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8370554, 1.8289304

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 6198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4586

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7069773, upper bound: 0.7086610
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7086290, upper bound: 0.7070083
time: 9.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8857856, 1.8594019
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4091573, 2.4108696
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6098685, 1.6262360
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9934831, 1.9945683
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0591121, 2.0498495
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7205315, 1.7058530
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.3988090, 2.3952751
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5405760, 2.5366039
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0878491, 2.0798049
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8291852, 1.8194029

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 6218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7069727, upper bound: 0.7131167
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7069727, upper bound: 0.7120884
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8838253, 1.8613596
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4066377, 2.4133892
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6116972, 1.6244087
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9918599, 1.9961920
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0574861, 2.0514750
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.7199621, 1.7064226
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.3968539, 2.3972313
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5402670, 2.5369148
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0862441, 2.0814104
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8306501, 1.8179386

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 457

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7084253, upper bound: 0.7151540
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7088860, upper bound: 0.7146921
time: 12.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8690672, 1.8960314
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4186230, 2.4123764
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6220131, 1.6103365
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9949961, 1.9918089
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0552149, 2.0637240
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6956196, 1.7050843
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.3993783, 2.3989539
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5118933, 2.5246053
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0784283, 2.0788212
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8211634, 1.8362124

Time for backsubstitution: 13.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7127028, upper bound: 0.7086851
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7124772, upper bound: 0.7089107
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -13.1174469, -10.4751902, -13.1174469, -10.4751902, -1.8693476, 1.8957508
1: -7.1292858, -4.1849308, -7.1292858, -4.1849308, -2.4189320, 2.4120693
2: 9.3677397, 11.2813492, 9.3677397, 11.2813492, -1.6266394, 1.6057103
3: -4.8719673, -2.7364025, -4.8719673, -2.7364025, -1.9957714, 1.9910336
4: -9.4387360, -6.7248473, -9.4387360, -6.7248473, -2.0555282, 2.0634108
5: -13.7978449, -11.1748800, -13.7978449, -11.1748800, -1.6885214, 1.7121828
6: -16.3375587, -12.7550831, -16.3375587, -12.7550831, -2.3974123, 2.4009202
7: -4.0563107, -1.3696804, -4.0563107, -1.3696804, -2.5204792, 2.5160198
8: -6.0375509, -3.6194940, -6.0375509, -3.6194940, -2.0720825, 2.0851669
9: -11.8428993, -9.3279104, -11.8428993, -9.3279104, -1.8231890, 1.8341868

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 457
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7120818, upper bound: 0.7044913
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7131103, upper bound: 0.7044892
time: 8.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 4586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7129210, upper bound: 0.7086662
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.7126954, upper bound: 0.7088920
time: 8.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 13.95 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.6408071517944336
rel_dist={2: [-0.7218570837425542, 0.7218564741351408]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 6218
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6198
type: RSZ, layer: 1, pos: 5762
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4586
type: RSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6087849, upper bound: 0.6071250
time: 9.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.6071229, upper bound: 0.6087845
time: 10.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.14 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 20.14
Output dim: 2, lower bound: -0.6087849, upper bound: 0.6071250
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 20.14
Output dim: 2, lower bound: -0.6071229, upper bound: 0.6087845
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.56638765335083
rel_dist={2: [-0.6118841057361983, 0.6118843662850555]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1692.07 seconds
