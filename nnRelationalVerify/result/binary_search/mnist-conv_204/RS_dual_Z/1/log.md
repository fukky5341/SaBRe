## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.4889943489
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.4903090, 2.4903090)
1: (-10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8988218, 2.8988218)
2: (-5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799)
3: (-12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1913986, 3.1913986)
4: (-8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297)
5: (-0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036)
6: (5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165)
7: (-18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.3238997, 3.3238983)
8: (-1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653)
9: (-8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.5020838, 2.5020838)

## BASE Result
execution time: IAR + LP analysis = 13.33 + 33.76 = 47.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.8584840, upper bound: 1.8584796


# Binary Search by BASE starts (time budget: 3552.91 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.309645652770996
rel_dist={6: [-1.4944306407619283, 1.4944305738519086]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2005128860473633
rel_dist={6: [-1.2600079034819816, 1.2600075251228153]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.2368907928466797
rel_dist={6: [-1.346308779708549, 1.3463093624116622]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=2.273268222808838
rel_dist={6: [-1.423624394997221, 1.4236241266449756]}

## Binary Search Result
Binary search time: 196.34 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 3356.57 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888994, upper bound: 1.6829872
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829876, upper bound: 1.6888992
time: 4.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.73
Output dim: 6, lower bound: -1.6888994, upper bound: 1.6829872
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.73
Output dim: 6, lower bound: -1.6829876, upper bound: 1.6888992

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3874850, 2.3931217
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8741522, 2.8719931
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1261692, 3.1261463
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3432875, 3.3484750
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0294089, 3.0412002
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4623718, 2.4682817

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6885491, upper bound: 1.6829739
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888860, upper bound: 1.6826357
time: 4.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3918552, 2.3874853
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8719931, 2.8736792
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1261463, 3.1261659
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3473077, 3.3432875
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0385489, 3.0294094
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4669509, 2.4623716

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6826364, upper bound: 1.6888856
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829743, upper bound: 1.6885485
time: 5.12 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 6, lower bound: -1.6885491, upper bound: 1.6829739
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 6, lower bound: -1.6888860, upper bound: 1.6826357
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 6, lower bound: -1.6826364, upper bound: 1.6888856
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 6, lower bound: -1.6829743, upper bound: 1.6885485

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3868260, 2.3945751
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8730936, 2.8742743
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1254005, 3.1278224
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3418918, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0280337, 3.0441999
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4611406, 2.4709694

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6885486, upper bound: 1.6823940
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6879738, upper bound: 1.6829747
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3874850, 2.3924627
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8741522, 2.8709354
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1261692, 3.1253777
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3432875, 3.3470798
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0294089, 3.0398250
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4623718, 2.4670503

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888856, upper bound: 1.6820605
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6883061, upper bound: 1.6826367
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3911943, 2.3889387
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8709345, 2.8759599
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1253777, 3.1278424
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3459120, 3.3463383
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0371737, 3.0324049
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4657178, 2.4650548

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6826360, upper bound: 1.6883055
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6820613, upper bound: 1.6888862
time: 4.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3918552, 2.3868260
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8719931, 2.8726211
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1261463, 3.1253972
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3473077, 3.3418922
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0385489, 3.0280342
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4669509, 2.4611404

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829739, upper bound: 1.6879733
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6823945, upper bound: 1.6885495
time: 4.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6885486, upper bound: 1.6823940
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6879738, upper bound: 1.6829747
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6888856, upper bound: 1.6820605
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6883061, upper bound: 1.6826367
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6826360, upper bound: 1.6883055
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6820613, upper bound: 1.6888862
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6829739, upper bound: 1.6879733
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.50
Output dim: 6, lower bound: -1.6823945, upper bound: 1.6885495

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3766112, 2.3782401
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8328986, 2.8491530
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1154461, 3.1219921
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3461351
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0347986, 3.0414352
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4301949, 2.4194877

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6885314, upper bound: 1.6780012
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6841558, upper bound: 1.6823766
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3704906, 2.3843603
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8479733, 2.8340783
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1195698, 3.1178679
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3364983, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0252686, 3.0509648
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4096584, 2.4400237

Time for backsubstitution: 13.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6879566, upper bound: 1.6785817
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6835810, upper bound: 1.6829574
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3772697, 2.3761272
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8339562, 2.8458142
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1162148, 3.1195474
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3416858
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0361748, 3.0370603
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4314265, 2.4155688

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888683, upper bound: 1.6776679
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6844928, upper bound: 1.6820434
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3711495, 2.3822477
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8490319, 2.8307395
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1203384, 3.1154232
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3378925, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0266447, 3.0465903
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4108906, 2.4361048

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882889, upper bound: 1.6782439
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6839133, upper bound: 1.6826193
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3809791, 2.3726034
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8307395, 2.8508391
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1154232, 3.1220112
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3409443
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0439396, 3.0296407
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4347720, 2.4135733

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6826187, upper bound: 1.6839126
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6782432, upper bound: 1.6882884
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3748589, 2.3787239
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8458142, 2.8357644
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1195469, 3.1178875
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3405161, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0344095, 3.0391703
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4142361, 2.4341092

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6820441, upper bound: 1.6844924
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6776685, upper bound: 1.6888678
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3816400, 2.3704908
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8317971, 2.8475003
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1161919, 3.1195664
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3364983
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0453148, 3.0252690
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4360056, 2.4096587

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829566, upper bound: 1.6835806
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6785811, upper bound: 1.6879561
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3755198, 2.3766112
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8468728, 2.8324256
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1203156, 3.1154428
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3419132, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0357857, 3.0347991
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4154701, 2.4301946

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6823774, upper bound: 1.6841554
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6780018, upper bound: 1.6885311
time: 4.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6885314, upper bound: 1.6780012
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6841558, upper bound: 1.6823766
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6879566, upper bound: 1.6785817
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6835810, upper bound: 1.6829574
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6888683, upper bound: 1.6776679
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6844928, upper bound: 1.6820434
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6882889, upper bound: 1.6782439
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6839133, upper bound: 1.6826193
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6826187, upper bound: 1.6839126
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6782432, upper bound: 1.6882884
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6820441, upper bound: 1.6844924
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6776685, upper bound: 1.6888678
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6829566, upper bound: 1.6835806
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6785811, upper bound: 1.6879561
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6823774, upper bound: 1.6841554
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.32
Output dim: 6, lower bound: -1.6780018, upper bound: 1.6885311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3750153, 2.3756833
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8329434, 2.8488364
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1190004, 3.1241941
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3341370, 3.3374605
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0281601, 3.0372725
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4294972, 2.4196751

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6735103, upper bound: 1.6779820
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6885112, upper bound: 1.6630731
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3740540, 2.3766441
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8325820, 2.8491974
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1176491, 3.1255460
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3419380, 3.3296595
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0306368, 3.0347962
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4303823, 2.4187901

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6691348, upper bound: 1.6823579
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6841356, upper bound: 1.6674711
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3688946, 2.3818035
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8480172, 2.8337622
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1231241, 3.1200705
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3200216, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0186300, 3.0468020
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4089608, 2.4402111

Time for backsubstitution: 13.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6730370, upper bound: 1.6785613
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6879364, upper bound: 1.6635764
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3679342, 2.3827643
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8476567, 2.8341227
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1217728, 3.1214218
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3278236, 3.3437743
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0211067, 3.0443258
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4098458, 2.4393260

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6686462, upper bound: 1.6829365
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6835608, upper bound: 1.6679518
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3756742, 2.3735704
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8340001, 2.8454981
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1197681, 3.1217494
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3355317, 3.3330112
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0295353, 3.0328975
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4307284, 2.4157560

Time for backsubstitution: 13.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6738628, upper bound: 1.6776487
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888482, upper bound: 1.6627331
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3747134, 2.3745313
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8336396, 2.8458586
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1184168, 3.1231012
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3433337, 3.3252096
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0320110, 3.0304213
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4316134, 2.4148710

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6694872, upper bound: 1.6820246
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6844726, upper bound: 1.6671242
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3695540, 2.3796909
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8490758, 2.8304234
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1238918, 3.1176252
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3214173, 3.3471260
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0200043, 3.0424271
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4101920, 2.4362919

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6733832, upper bound: 1.6782235
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882688, upper bound: 1.6632229
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3685927, 2.3806517
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8487153, 2.8307838
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1225405, 3.1189771
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3292184, 3.3393245
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0224819, 3.0399513
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4110775, 2.4354069

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6689853, upper bound: 1.6825990
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6838932, upper bound: 1.6675985
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3793831, 2.3700466
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8307843, 2.8505225
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1189775, 3.1242132
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3381553, 3.3322697
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0373001, 3.0254774
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4340749, 2.4137604

Time for backsubstitution: 15.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6675978, upper bound: 1.6838936
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6825986, upper bound: 1.6689846
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3784227, 2.3710074
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8304229, 2.8508835
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1176262, 3.1255651
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3459573, 3.3244677
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0397758, 3.0230017
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4349599, 2.4128754

Time for backsubstitution: 15.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6632222, upper bound: 1.6882696
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6782230, upper bound: 1.6733826
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3732624, 2.3761671
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8458581, 2.8354483
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1231012, 3.1200895
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3240409, 3.3463845
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0277710, 3.0350075
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4135394, 2.4342964

Time for backsubstitution: 15.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6671245, upper bound: 1.6844730
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6820239, upper bound: 1.6694864
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3723021, 2.3771279
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8454976, 2.8358088
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1217499, 3.1214409
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3318419, 3.3385830
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0302467, 3.0325313
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4144244, 2.4334114

Time for backsubstitution: 15.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6627335, upper bound: 1.6888486
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6776483, upper bound: 1.6738620
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3800440, 2.3679340
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8318410, 2.8471842
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1197453, 3.1217685
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3395524, 3.3278236
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0386744, 3.0211067
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4353080, 2.4098461

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6679512, upper bound: 1.6835614
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829365, upper bound: 1.6686457
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3790827, 2.3688948
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8314805, 2.8475447
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1183939, 3.1231203
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3473535, 3.3200221
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0411510, 3.0186300
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4361930, 2.4089611

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6635756, upper bound: 1.6879371
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6785609, upper bound: 1.6730364
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3739233, 2.3740544
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8469167, 2.8321095
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1238689, 3.1176443
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3254371, 3.3419385
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0291452, 3.0306363
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4147716, 2.4303820

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6674716, upper bound: 1.6841360
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6823572, upper bound: 1.6691341
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3729620, 2.3750153
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8465562, 2.8324699
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1225176, 3.1189961
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3332391, 3.3341370
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0316210, 3.0281601
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4156566, 2.4294970

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6630735, upper bound: 1.6885116
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6779816, upper bound: 1.6735093
time: 4.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6735103, upper bound: 1.6779820
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6885112, upper bound: 1.6630731
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6691348, upper bound: 1.6823579
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6841356, upper bound: 1.6674711
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6730370, upper bound: 1.6785613
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6879364, upper bound: 1.6635764
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6686462, upper bound: 1.6829365
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6835608, upper bound: 1.6679518
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6738628, upper bound: 1.6776487
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6888482, upper bound: 1.6627331
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6694872, upper bound: 1.6820246
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6844726, upper bound: 1.6671242
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6733832, upper bound: 1.6782235
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6882688, upper bound: 1.6632229
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6689853, upper bound: 1.6825990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6838932, upper bound: 1.6675985
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6675978, upper bound: 1.6838936
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6825986, upper bound: 1.6689846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6632222, upper bound: 1.6882696
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6782230, upper bound: 1.6733826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6671245, upper bound: 1.6844730
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6820239, upper bound: 1.6694864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6627335, upper bound: 1.6888486
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6776483, upper bound: 1.6738620
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6679512, upper bound: 1.6835614
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6829365, upper bound: 1.6686457
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6635756, upper bound: 1.6879371
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6785609, upper bound: 1.6730364
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6674716, upper bound: 1.6841360
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6823572, upper bound: 1.6691341
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6630735, upper bound: 1.6885116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.55
Output dim: 6, lower bound: -1.6779816, upper bound: 1.6735093

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3789873, 2.3642251
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8341918, 2.8452435
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1198373, 3.1217403
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3403077, 3.3196688
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0232086, 3.0389709
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4355288, 2.4022126

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6734958, upper bound: 1.6778830
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6734114, upper bound: 1.6779678
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3635569, 2.3756833
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8293500, 2.8488364
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1165462, 3.1241941
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3163447, 3.3374605
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0281601, 3.0323219
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4120345, 2.4196751

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6884965, upper bound: 1.6629743
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6884121, upper bound: 1.6630589
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3780260, 2.3651860
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8338313, 2.8456044
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1184850, 3.1230917
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3481088, 3.3118677
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0256863, 3.0364943
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4364138, 2.4013276

Time for backsubstitution: 12.81 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.363316535949707
rel_dist={6: [-1.6889095697923429, 1.6889090371909665]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5621023, upper bound: 1.5564134
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5564127, upper bound: 1.5621029
time: 4.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.91
Output dim: 6, lower bound: -1.5621023, upper bound: 1.5564134
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.91
Output dim: 6, lower bound: -1.5564127, upper bound: 1.5621029

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2635889, 2.2679729
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7239294, 2.7222500
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9847975, 2.9847789
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1768608, 3.1808958
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3457603, 2.3460243
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8391767, 2.8483472
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3357735, 2.3403697

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617585, upper bound: 1.5563995
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620883, upper bound: 1.5560679
time: 4.24 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2679591, 2.2635889
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7222490, 2.7239361
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9847784, 2.9847984
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1808810, 3.1768608
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3460236, 2.3457601
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8483176, 2.8391762
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3403525, 2.3357730

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560674, upper bound: 1.5620890
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5563989, upper bound: 1.5617591
time: 4.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 6, lower bound: -1.5617585, upper bound: 1.5563995
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 6, lower bound: -1.5620883, upper bound: 1.5560679
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 6, lower bound: -1.5560674, upper bound: 1.5620890
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 6, lower bound: -1.5563989, upper bound: 1.5617591

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2629299, 2.2689569
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7228708, 2.7237887
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9840288, 2.9859118
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1754661, 3.1829615
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3455992, 2.3462589
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8378015, 2.8503747
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3345423, 2.3421865

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617582, upper bound: 1.5561149
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5614787, upper bound: 1.5563991
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2635889, 2.2673137
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7239294, 2.7211919
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9847975, 2.9840102
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1768608, 3.1795006
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3457603, 2.3458631
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8391767, 2.8469720
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3357735, 2.3391383

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620880, upper bound: 1.5557855
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5618041, upper bound: 1.5560675
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2672982, 2.2645729
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7211914, 2.7254744
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9840097, 2.9859314
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1794863, 3.1789236
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3458619, 2.3459947
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8469424, 2.8412008
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3391190, 2.3375862

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560671, upper bound: 1.5618043
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5557850, upper bound: 1.5620885
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2679591, 2.2629297
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7222490, 2.7228775
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9847784, 2.9840298
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1808810, 3.1754656
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3460236, 2.3455989
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8483176, 2.8378010
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3403525, 2.3345416

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5563987, upper bound: 1.5614793
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5561144, upper bound: 1.5617584
time: 4.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5617582, upper bound: 1.5561149
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5614787, upper bound: 1.5563991
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5620880, upper bound: 1.5557855
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5618041, upper bound: 1.5560675
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5560671, upper bound: 1.5618043
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5557850, upper bound: 1.5620885
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5563987, upper bound: 1.5614793
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 6, lower bound: -1.5561144, upper bound: 1.5617584

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2513547, 2.2526217
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6826758, 2.6953177
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9740734, 2.9791646
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1810493, 3.1775675
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3472352, 2.3495450
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8424482, 2.8476100
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2990322, 2.2907047

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617448, upper bound: 1.5527169
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583599, upper bound: 1.5561016
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2465940, 2.2573819
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6944003, 2.6835933
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9772816, 2.9759574
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1700716, 3.1885452
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3488851, 2.3478951
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8350363, 2.8550220
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2830601, 2.3066773

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5614653, upper bound: 1.5530004
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5580804, upper bound: 1.5563854
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2520137, 2.2509785
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6837335, 2.6927214
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9748421, 2.9772630
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1824446, 3.1741066
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3473964, 2.3491488
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8438244, 2.8442073
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3002644, 2.2876568

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620746, upper bound: 1.5523873
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5586898, upper bound: 1.5557722
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2472529, 2.2557387
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6954579, 2.6809964
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780502, 2.9740558
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1714659, 3.1850843
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3490462, 2.3474994
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8364124, 2.8516192
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2842917, 2.3036292

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617907, upper bound: 1.5526691
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5584058, upper bound: 1.5560537
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2557230, 2.2482376
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6809964, 2.6970038
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9740562, 2.9791842
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1850681, 3.1735296
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3474994, 2.3492808
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8515892, 2.8384361
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3036094, 2.2861047

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560537, upper bound: 1.5584065
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5526688, upper bound: 1.5617912
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2509623, 2.2529979
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6927209, 2.6852794
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9772635, 2.9759769
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1740894, 3.1845078
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3491492, 2.3476310
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8441772, 2.8458486
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2876372, 2.3020771

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5557716, upper bound: 1.5586900
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5523867, upper bound: 1.5620749
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2563839, 2.2465944
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6820540, 2.6944075
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9748249, 2.9772825
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1864653, 3.1700716
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3476601, 2.3488851
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8529644, 2.8350368
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3048439, 2.2830601

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5563852, upper bound: 1.5580811
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5530004, upper bound: 1.5614660
time: 4.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2516232, 2.2513547
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6937795, 2.6826825
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780321, 2.9740753
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1754866, 3.1810498
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3493099, 2.3472352
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8455524, 2.8424487
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2888718, 2.2990324

Time for backsubstitution: 13.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5561010, upper bound: 1.5583602
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5527162, upper bound: 1.5617452
time: 4.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5617448, upper bound: 1.5527169
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5583599, upper bound: 1.5561016
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5614653, upper bound: 1.5530004
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5580804, upper bound: 1.5563854
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5620746, upper bound: 1.5523873
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5586898, upper bound: 1.5557722
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5617907, upper bound: 1.5526691
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5584058, upper bound: 1.5560537
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5560537, upper bound: 1.5584065
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5526688, upper bound: 1.5617912
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5557716, upper bound: 1.5586900
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5523867, upper bound: 1.5620749
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5563852, upper bound: 1.5580811
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5530004, upper bound: 1.5614660
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5561010, upper bound: 1.5583602
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.89
Output dim: 6, lower bound: -1.5527162, upper bound: 1.5617452

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2495451, 2.2500648
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6826396, 2.6950016
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9773273, 2.9813671
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1645737, 3.1671591
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3460283, 2.3478427
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8358097, 2.8428969
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2983346, 2.2906954

Time for backsubstitution: 13.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5467099, upper bound: 1.5526972
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617247, upper bound: 1.5376888
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2487974, 2.2508121
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6823592, 2.6952820
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9762764, 2.9824181
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1706419, 3.1610909
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3455338, 2.3483372
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8377352, 2.8409710
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2990232, 2.2900071

Time for backsubstitution: 13.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5433250, upper bound: 1.5560821
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583399, upper bound: 1.5410735
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2447844, 2.2548251
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6943650, 2.6832767
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9805346, 2.9781599
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1535950, 3.1781373
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3476777, 2.3461933
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8283978, 2.8503094
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2823625, 2.3066678

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5464326, upper bound: 1.5529810
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5614456, upper bound: 1.5379658
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2440376, 2.2555723
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6940837, 2.6835575
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9794836, 2.9792109
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1596632, 3.1720695
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3471828, 2.3466878
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8303232, 2.8483829
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2830510, 2.3059795

Time for backsubstitution: 12.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5430477, upper bound: 1.5563661
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5580608, upper bound: 1.5413506
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2502046, 2.2484217
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6836972, 2.6924047
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780960, 2.9794655
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1659694, 3.1636982
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3461895, 2.3474469
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8371849, 2.8394942
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2995658, 2.2876475

Time for backsubstitution: 13.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5470397, upper bound: 1.5523677
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620544, upper bound: 1.5373568
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2494569, 2.2491689
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6834168, 2.6926851
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9770441, 2.9805164
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1720366, 3.1576300
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3456950, 2.3479414
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8391113, 2.8375683
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3002543, 2.2869589

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5436549, upper bound: 1.5557528
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5586696, upper bound: 1.5407416
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2454438, 2.2531819
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6954217, 2.6806798
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9813032, 2.9762583
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1549907, 3.1746764
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3478389, 2.3457975
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8297729, 2.8469067
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2835937, 2.3036196

Time for backsubstitution: 13.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5467624, upper bound: 1.5526493
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617711, upper bound: 1.5376350
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2446966, 2.2539291
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6951413, 2.6809607
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9802513, 2.9773092
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1610579, 3.1686087
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3473439, 2.3462920
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8316975, 2.8449802
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2842822, 2.3029315

Time for backsubstitution: 13.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5433776, upper bound: 1.5560346
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583862, upper bound: 1.5410199
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2539129, 2.2456808
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6809602, 2.6966877
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9773092, 2.9813862
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1685920, 3.1631212
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3462915, 2.3475785
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8449497, 2.8337231
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3029132, 2.2860954

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5410194, upper bound: 1.5583869
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560337, upper bound: 1.5433782
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2531662, 2.2464280
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6806798, 2.6969681
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9762583, 2.9824371
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1746602, 3.1570535
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3457966, 2.3480730
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8468761, 2.8317971
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3036008, 2.2854068

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5376346, upper bound: 1.5617717
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5526488, upper bound: 1.5467632
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2491531, 2.2504411
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6926856, 2.6849627
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9805164, 2.9781790
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1576142, 3.1740994
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3479409, 2.3459291
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8375378, 2.8411350
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2869401, 2.3020675

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5407412, upper bound: 1.5586702
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5557520, upper bound: 1.5436550
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2484055, 2.2511885
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6924043, 2.6852436
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9794655, 2.9792299
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1636815, 3.1680317
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3474464, 2.3464236
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8394642, 2.8392096
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2876287, 2.3013794

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5373563, upper bound: 1.5620550
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5523672, upper bound: 1.5470400
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2545738, 2.2440376
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6820188, 2.6940908
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780779, 2.9794846
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1699891, 3.1596632
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3464537, 2.3471832
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8463240, 2.8303232
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3041453, 2.2830508

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5413503, upper bound: 1.5580615
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5563652, upper bound: 1.5430483
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2538261, 2.2447848
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6817384, 2.6943712
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9770260, 2.9805355
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1760573, 3.1535954
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3459592, 2.3476772
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8482504, 2.8283978
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3048339, 2.2823622

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5379655, upper bound: 1.5614462
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5529803, upper bound: 1.5464332
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2498131, 2.2487979
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6937432, 2.6823659
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9812851, 2.9762774
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1590104, 3.1706419
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3481030, 2.3455334
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8389120, 2.8377357
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2881732, 2.2990229

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5410730, upper bound: 1.5583405
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560815, upper bound: 1.5433253
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2490664, 2.2495453
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6934628, 2.6826468
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9802341, 2.9773283
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1650786, 3.1645737
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3476086, 2.3460283
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8408384, 2.8358097
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2888618, 2.2983346

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5376882, upper bound: 1.5617253
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5526966, upper bound: 1.5467101
time: 4.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5467099, upper bound: 1.5526972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5617247, upper bound: 1.5376888
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5433250, upper bound: 1.5560821
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5583399, upper bound: 1.5410735
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5464326, upper bound: 1.5529810
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5614456, upper bound: 1.5379658
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5430477, upper bound: 1.5563661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5580608, upper bound: 1.5413506
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5470397, upper bound: 1.5523677
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5620544, upper bound: 1.5373568
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5436549, upper bound: 1.5557528
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5586696, upper bound: 1.5407416
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5467624, upper bound: 1.5526493
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5617711, upper bound: 1.5376350
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5433776, upper bound: 1.5560346
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5583862, upper bound: 1.5410199
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5410194, upper bound: 1.5583869
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5560337, upper bound: 1.5433782
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5376346, upper bound: 1.5617717
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5526488, upper bound: 1.5467632
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5407412, upper bound: 1.5586702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5557520, upper bound: 1.5436550
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5373563, upper bound: 1.5620550
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5523672, upper bound: 1.5470400
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5413503, upper bound: 1.5580615
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5563652, upper bound: 1.5430483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5379655, upper bound: 1.5614462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5529803, upper bound: 1.5464332
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5410730, upper bound: 1.5583405
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5560815, upper bound: 1.5433253
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5376882, upper bound: 1.5617253
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.69
Output dim: 6, lower bound: -1.5526966, upper bound: 1.5467101

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2500887, 2.2386069
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6828122, 2.6914082
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9774327, 2.9789128
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1654191, 3.1493673
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3355041, 2.3483386
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8308582, 2.8431177
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2991452, 2.2732329

Time for backsubstitution: 12.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5466965, upper bound: 1.5526002
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5466125, upper bound: 1.5526845
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2380877, 2.2500648
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6790471, 2.6950016
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9748731, 2.9813671
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1467814, 3.1671591
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3460283, 2.3373184
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8358097, 2.8379469
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2808723, 2.2906954

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617113, upper bound: 1.5375918
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5616273, upper bound: 1.5376758
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2493410, 2.2393541
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6825318, 2.6916890
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9763818, 2.9799643
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1714864, 3.1432991
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3350091, 2.3488336
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8327847, 2.8411922
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2998338, 2.2725446

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5433116, upper bound: 1.5559850
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5432276, upper bound: 1.5560692
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2373400, 2.2508121
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6787667, 2.6952820
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9738221, 2.9824181
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1528497, 3.1610909
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3455338, 2.3378129
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8377352, 2.8360205
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2815609, 2.2900071

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583265, upper bound: 1.5409765
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5582425, upper bound: 1.5410609
time: 4.44 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5466965, upper bound: 1.5526002
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5466125, upper bound: 1.5526845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5617113, upper bound: 1.5375918
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5616273, upper bound: 1.5376758
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5433116, upper bound: 1.5559850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5432276, upper bound: 1.5560692
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5583265, upper bound: 1.5409765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.64
Output dim: 6, lower bound: -1.5582425, upper bound: 1.5410609
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5464326, upper bound: 1.5529810
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5614456, upper bound: 1.5379658
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5430477, upper bound: 1.5563661
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5580608, upper bound: 1.5413506
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5470397, upper bound: 1.5523677
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5620544, upper bound: 1.5373568
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5436549, upper bound: 1.5557528
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5586696, upper bound: 1.5407416
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5467624, upper bound: 1.5526493
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5617711, upper bound: 1.5376350
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5433776, upper bound: 1.5560346
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5583862, upper bound: 1.5410199
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5410194, upper bound: 1.5583869
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5560337, upper bound: 1.5433782
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5376346, upper bound: 1.5617717
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5526488, upper bound: 1.5467632
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5407412, upper bound: 1.5586702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5557520, upper bound: 1.5436550
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5373563, upper bound: 1.5620550
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5523672, upper bound: 1.5470400
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5413503, upper bound: 1.5580615
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5563652, upper bound: 1.5430483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5379655, upper bound: 1.5614462
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5529803, upper bound: 1.5464332
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5410730, upper bound: 1.5583405
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5560815, upper bound: 1.5433253
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5376882, upper bound: 1.5617253
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5526966, upper bound: 1.5467101
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.3460235595703125
rel_dist={6: [-1.5621134080175434, 1.562113956458468]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944223, upper bound: 1.4895114
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4895114, upper bound: 1.4944224
time: 4.23 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.78
Output dim: 6, lower bound: -1.4944223, upper bound: 1.4895114
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.78
Output dim: 6, lower bound: -1.4895114, upper bound: 1.4944224

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2016406, 2.2053986
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6488180, 2.6473784
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9141111, 2.9140949
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0936480, 3.0971060
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3093824, 2.3096087
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7440600, 2.7519207
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2724733, 2.2764137

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941052, upper bound: 1.4894974
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944085, upper bound: 1.4891928
time: 4.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2053986, 2.2016408
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6473780, 2.6488175
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9140959, 2.9141102
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0971060, 3.0936480
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3096089, 2.3093827
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7519212, 2.7440600
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2764139, 2.2724736

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891932, upper bound: 1.4944085
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4894974, upper bound: 1.4941048
time: 4.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 6, lower bound: -1.4941052, upper bound: 1.4894974
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 6, lower bound: -1.4944085, upper bound: 1.4891928
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 6, lower bound: -1.4891932, upper bound: 1.4944085
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.98
Output dim: 6, lower bound: -1.4894974, upper bound: 1.4941048

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2009816, 2.2061477
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6477594, 2.6485462
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9133425, 2.9149566
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0922523, 3.0986772
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3092213, 2.3097870
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7426848, 2.7534618
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2712421, 2.2777951

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941049, upper bound: 1.4893173
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4939112, upper bound: 1.4894975
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2016406, 2.2047393
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6488180, 2.6463203
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9141111, 2.9133267
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0936480, 3.0957108
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3093824, 2.3094475
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7440600, 2.7505455
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2724733, 2.2751822

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944082, upper bound: 1.4890176
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4942099, upper bound: 1.4891932
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2047396, 2.2023900
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6463203, 2.6499863
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9133272, 2.9149728
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0957103, 3.0952163
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3094478, 2.3095605
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7505460, 2.7455988
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2751827, 2.2738519

Time for backsubstitution: 12.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891929, upper bound: 1.4942098
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4890175, upper bound: 1.4944086
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2053986, 2.2009816
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6473780, 2.6477599
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9140959, 2.9133415
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0971060, 3.0922527
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3096089, 2.3092215
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7519212, 2.7426848
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2764139, 2.2712424

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4894972, upper bound: 1.4939113
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4893172, upper bound: 1.4941052
time: 5.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4941049, upper bound: 1.4893173
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4939112, upper bound: 1.4894975
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4944082, upper bound: 1.4890176
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4942099, upper bound: 1.4891932
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4891929, upper bound: 1.4942098
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4890175, upper bound: 1.4944086
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4894972, upper bound: 1.4939113
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.98
Output dim: 6, lower bound: -1.4893172, upper bound: 1.4941052

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1887259, 2.1898124
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6075644, 2.6184006
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9033871, 2.9077516
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0962677, 3.0932832
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3108578, 2.3128371
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7462730, 2.7506976
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2334514, 2.2263134

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940930, upper bound: 1.4864023
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4911913, upper bound: 1.4893053
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1846461, 2.1938927
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6176143, 2.6083508
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9061375, 2.9050021
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0868587, 3.1026931
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3122716, 2.3114233
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7399197, 2.7570505
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2197609, 2.2400041

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4938992, upper bound: 1.4865839
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4909963, upper bound: 1.4894854
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1893849, 2.1884041
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6086221, 2.6161747
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9041557, 2.9061213
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0976629, 3.0903168
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3110189, 2.3124976
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7476492, 2.7477808
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2346830, 2.2237008

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943963, upper bound: 1.4861024
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4914946, upper bound: 1.4890057
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1853051, 2.1924844
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6186719, 2.6061249
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9069061, 2.9033723
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0882530, 3.0997267
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3124328, 2.3110843
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7412958, 2.7541342
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2209926, 2.2373915

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941979, upper bound: 1.4862795
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912950, upper bound: 1.4891812
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1924844, 2.1860547
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6061254, 2.6198401
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9033718, 2.9077678
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0997267, 3.0898223
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3110843, 2.3126106
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7541342, 2.7428341
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2373915, 2.2223704

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891810, upper bound: 1.4912951
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4862793, upper bound: 1.4941981
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1884036, 2.1901350
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6161752, 2.6097908
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9061222, 2.9050183
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0903168, 3.0992322
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3124976, 2.3111968
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7477808, 2.7491875
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2237005, 2.2360611

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4890055, upper bound: 1.4914949
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4861026, upper bound: 1.4943962
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1931429, 2.1846464
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6071830, 2.6176143
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9041414, 2.9061365
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1011209, 3.0868583
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3112454, 2.3122716
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7555103, 2.7399201
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2386231, 2.2197607

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4894852, upper bound: 1.4909960
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4865835, upper bound: 1.4938994
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1890626, 2.1887267
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6172328, 2.6075644
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9068909, 2.9033875
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0917110, 3.0962682
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3126588, 2.3108578
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7491570, 2.7462735
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2249327, 2.2334514

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4893052, upper bound: 1.4911918
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4864023, upper bound: 1.4940932
time: 4.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4940930, upper bound: 1.4864023
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4911913, upper bound: 1.4893053
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4938992, upper bound: 1.4865839
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4909963, upper bound: 1.4894854
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4943963, upper bound: 1.4861024
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4914946, upper bound: 1.4890057
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4941979, upper bound: 1.4862795
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4912950, upper bound: 1.4891812
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4891810, upper bound: 1.4912951
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4862793, upper bound: 1.4941981
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4890055, upper bound: 1.4914949
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4861026, upper bound: 1.4943962
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4894852, upper bound: 1.4909960
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4865835, upper bound: 1.4938994
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4893052, upper bound: 1.4911918
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.37
Output dim: 6, lower bound: -1.4864023, upper bound: 1.4940932

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1868105, 2.1872556
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6074882, 2.6180840
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9064903, 2.9099531
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0797920, 3.0820079
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3095798, 2.3111353
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7396345, 2.7457094
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2327533, 2.2262056

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4806218, upper bound: 1.4863913
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940817, upper bound: 1.4729369
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1861696, 2.1878963
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6072478, 2.6183243
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9055901, 2.9108543
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0849934, 3.0768070
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3091559, 2.3115592
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7412853, 2.7440586
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2333436, 2.2256157

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4777061, upper bound: 1.4892937
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4911799, upper bound: 1.4758391
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1827297, 2.1913359
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6175380, 2.6080341
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9092398, 2.9072042
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0703821, 3.0914178
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3109937, 2.3097215
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7332811, 2.7520623
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2190633, 2.2398963

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4804315, upper bound: 1.4865724
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4938879, upper bound: 1.4731086
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1820898, 2.1919765
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6172976, 2.6082745
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9083385, 2.9081054
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0755825, 3.0862169
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3105698, 2.3101454
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7349319, 2.7504115
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2196536, 2.2393062

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4775293, upper bound: 1.4894742
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4909850, upper bound: 1.4760195
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1874695, 2.1858473
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6085458, 2.6158581
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9072590, 2.9083233
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0811877, 3.0790415
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3097410, 2.3107958
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7410088, 2.7427926
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2339849, 2.2235930

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4809254, upper bound: 1.4860912
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943850, upper bound: 1.4726358
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1868291, 2.1864877
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6083055, 2.6160984
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9063578, 2.9092245
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0863881, 3.0738406
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3093171, 2.3112202
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7426605, 2.7411418
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2345748, 2.2230029

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4780100, upper bound: 1.4889940
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4914832, upper bound: 1.4755381
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1833892, 2.1899276
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6185956, 2.6058083
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9100084, 2.9055743
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0717769, 3.0884514
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3111548, 2.3093824
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7346554, 2.7491460
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2202945, 2.2372837

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4807316, upper bound: 1.4862684
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941867, upper bound: 1.4728038
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1827483, 2.1905680
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6183553, 2.6060486
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9091072, 2.9064755
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0769782, 3.0832505
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3107309, 2.3098059
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7363071, 2.7474952
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2208843, 2.2366936

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4778294, upper bound: 1.4891700
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912837, upper bound: 1.4757150
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1905680, 2.1834979
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6060491, 2.6195240
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9064751, 2.9099698
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0832500, 3.0785470
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3098063, 2.3109088
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7474947, 2.7378464
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2366939, 2.2222626

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4757148, upper bound: 1.4912839
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891696, upper bound: 1.4778296
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1899271, 2.1841385
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6058087, 2.6197643
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9055748, 2.9108710
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0884514, 3.0733461
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3093824, 2.3113327
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7491465, 2.7361951
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2372842, 2.2216725

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4728034, upper bound: 1.4941869
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4862679, upper bound: 1.4807320
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1864872, 2.1875782
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6160989, 2.6094742
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9092245, 2.9072208
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0738401, 3.0879569
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3112197, 2.3094950
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7411413, 2.7441993
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2230029, 2.2359531

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4755377, upper bound: 1.4914835
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4889942, upper bound: 1.4780102
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1858473, 2.1882188
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6158586, 2.6097145
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9083233, 2.9081216
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0790415, 3.0827560
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3107963, 2.3099189
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7427931, 2.7425485
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2235932, 2.2353632

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4726354, upper bound: 1.4943852
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4860913, upper bound: 1.4809258
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1912270, 2.1820896
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6071057, 2.6172972
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9072437, 2.9083385
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0846457, 3.0755835
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3099675, 2.3105698
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7488708, 2.7349319
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2379251, 2.2196529

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4760192, upper bound: 1.4909854
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4894739, upper bound: 1.4775297
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1905866, 2.1827302
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6068654, 2.6175380
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9063425, 2.9092398
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0898471, 3.0703821
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3095436, 2.3109937
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7505207, 2.7332811
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2385149, 2.2190630

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4731082, upper bound: 1.4938878
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4865722, upper bound: 1.4804318
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1871467, 2.1861699
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6171556, 2.6072478
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9099932, 2.9055896
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0752358, 3.0849934
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3113809, 2.3091559
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7425175, 2.7412853
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2242341, 2.2333436

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4758387, upper bound: 1.4911803
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4892940, upper bound: 1.4777066
time: 4.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1865063, 2.1868105
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6169152, 2.6074882
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9090919, 2.9064903
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0804362, 3.0797920
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3109574, 2.3095798
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7441673, 2.7396345
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2248244, 2.2327535

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729365, upper bound: 1.4940820
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4863910, upper bound: 1.4806222
time: 4.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4806218, upper bound: 1.4863913
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4940817, upper bound: 1.4729369
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4777061, upper bound: 1.4892937
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4911799, upper bound: 1.4758391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4804315, upper bound: 1.4865724
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4938879, upper bound: 1.4731086
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4775293, upper bound: 1.4894742
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4909850, upper bound: 1.4760195
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4809254, upper bound: 1.4860912
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4943850, upper bound: 1.4726358
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4780100, upper bound: 1.4889940
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4914832, upper bound: 1.4755381
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4807316, upper bound: 1.4862684
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4941867, upper bound: 1.4728038
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4778294, upper bound: 1.4891700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4912837, upper bound: 1.4757150
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4757148, upper bound: 1.4912839
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4891696, upper bound: 1.4778296
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4728034, upper bound: 1.4941869
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4862679, upper bound: 1.4807320
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4755377, upper bound: 1.4914835
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4889942, upper bound: 1.4780102
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4726354, upper bound: 1.4943852
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4860913, upper bound: 1.4809258
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4760192, upper bound: 1.4909854
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4894739, upper bound: 1.4775297
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4731082, upper bound: 1.4938878
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4865722, upper bound: 1.4804318
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4758387, upper bound: 1.4911803
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4892940, upper bound: 1.4777066
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4729365, upper bound: 1.4940820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 21.79
Output dim: 6, lower bound: -1.4863910, upper bound: 1.4806222

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1753521, 2.1860843
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6038957, 2.6177187
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9040370, 2.9096932
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0619998, 3.0801907
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3085017, 2.3006110
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7391167, 2.7407589
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2152910, 2.2244058

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940691, upper bound: 1.4728406
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4939853, upper bound: 1.4729248
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1849985, 2.1764381
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6068826, 2.6147313
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9053292, 2.9084005
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0831761, 3.0590153
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.2986317, 2.3104806
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7363338, 2.7435408
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2315435, 2.2081532

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4776937, upper bound: 1.4891980
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4776099, upper bound: 1.4892823
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1747122, 2.1867247
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6036544, 2.6179590
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9031358, 2.9105945
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0672011, 3.0749898
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3080778, 2.3010349
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7407665, 2.7391081
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2158813, 2.2238157

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4911674, upper bound: 1.4757424
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4910836, upper bound: 1.4758269
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1712723, 2.1901646
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6139455, 2.6076689
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9067864, 2.9069443
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0525899, 3.0896010
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3099155, 2.2991972
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7327633, 2.7471123
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2016006, 2.2380962

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4938756, upper bound: 1.4730119
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4937918, upper bound: 1.4730964
time: 4.06 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4940691, upper bound: 1.4728406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4939853, upper bound: 1.4729248
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4776937, upper bound: 1.4891980
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4776099, upper bound: 1.4892823
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4911674, upper bound: 1.4757424
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4910836, upper bound: 1.4758269
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4938756, upper bound: 1.4730119
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -1.4937918, upper bound: 1.4730964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4775293, upper bound: 1.4894742
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4909850, upper bound: 1.4760195
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4943850, upper bound: 1.4726358
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4914832, upper bound: 1.4755381
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4941867, upper bound: 1.4728038
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4778294, upper bound: 1.4891700
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4912837, upper bound: 1.4757150
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4757148, upper bound: 1.4912839
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4891696, upper bound: 1.4778296
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4728034, upper bound: 1.4941869
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4755377, upper bound: 1.4914835
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4726354, upper bound: 1.4943852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4760192, upper bound: 1.4909854
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4894739, upper bound: 1.4775297
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4731082, upper bound: 1.4938878
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4758387, upper bound: 1.4911803
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4892940, upper bound: 1.4777066
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -1.4729365, upper bound: 1.4940820
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.309645652770996
rel_dist={6: [-1.4944306407619283, 1.4944305738519086]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2418.95 seconds
