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
execution time: IAR + LP analysis = 13.51 + 34.19 = 47.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.8584840, upper bound: 1.8584796


# Binary Search by BASE starts (time budget: 3552.30 seconds, max iter: 100)

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
Binary search time: 197.37 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 3354.93 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6889080, upper bound: 1.6882565
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882573, upper bound: 1.6889073
time: 4.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.92
Output dim: 6, lower bound: -1.6889080, upper bound: 1.6882565
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.92
Output dim: 6, lower bound: -1.6882573, upper bound: 1.6889073

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3913527, 2.3910520
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8860974, 2.8907537
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1244674, 3.1268210
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3494816, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0220346, 3.0282083
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4653797, 2.4644372

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6887891, upper bound: 1.6882515
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6889030, upper bound: 1.6881377
time: 4.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3910518, 2.3913531
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8907542, 2.8860974
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1268210, 3.1244674
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3494811
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0282087, 3.0220342
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4644375, 2.4653797

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882430, upper bound: 1.6888086
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6881584, upper bound: 1.6888935
time: 4.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 6, lower bound: -1.6887891, upper bound: 1.6882515
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 6, lower bound: -1.6889030, upper bound: 1.6881377
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 6, lower bound: -1.6882430, upper bound: 1.6888086
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.28
Output dim: 6, lower bound: -1.6881584, upper bound: 1.6888935

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3907986, 2.3981295
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8856702, 2.8962011
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1236634, 3.1370769
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0213709, 3.0366631
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4672036, 2.4642942

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6887749, upper bound: 1.6881529
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6886902, upper bound: 1.6882373
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3913527, 2.3904979
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8860974, 2.8903270
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1244674, 3.1260171
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3493834, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0220346, 3.0275445
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4652371, 2.4644372

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888928, upper bound: 1.6822159
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829811, upper bound: 1.6881277
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3907194, 2.3950856
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8908854, 2.8860979
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1266294, 3.1266203
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0281725, 3.0224366
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4640293, 2.4699559

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6768954, upper bound: 1.6788841
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6783222, upper bound: 1.6774568
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3910518, 2.3910208
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8907547, 2.8860974
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1268210, 3.1242752
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3494458
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0282087, 3.0219989
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4644375, 2.4649723

Time for backsubstitution: 13.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6881411, upper bound: 1.6845007
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6837655, upper bound: 1.6888759
time: 4.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6887749, upper bound: 1.6881529
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6886902, upper bound: 1.6882373
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6888928, upper bound: 1.6822159
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6829811, upper bound: 1.6881277
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6768954, upper bound: 1.6788841
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6783222, upper bound: 1.6774568
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6881411, upper bound: 1.6845007
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.24
Output dim: 6, lower bound: -1.6837655, upper bound: 1.6888759

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3904667, 2.4018626
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8858013, 2.8962011
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1234717, 3.1392303
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0213366, 3.0370655
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4667964, 2.4688702

Time for backsubstitution: 12.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6887577, upper bound: 1.6837600
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6843821, upper bound: 1.6881356
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3907986, 2.3977976
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8856707, 2.8962011
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1236634, 3.1368852
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0213709, 3.0366273
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4672036, 2.4638865

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6883393, upper bound: 1.6882244
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6886773, upper bound: 1.6878862
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3869839, 2.3917651
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8865705, 2.8886404
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1244712, 3.1259980
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3453631, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0128937, 3.0301957
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4606571, 2.4657681

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6738861, upper bound: 1.6821968
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888727, upper bound: 1.6672289
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3913527, 2.3861284
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8844113, 2.8903270
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1244483, 3.1260171
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3493834, 3.3464999
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0220346, 3.0184050
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4652371, 2.4598579

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6829669, upper bound: 1.6880285
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6828822, upper bound: 1.6881132
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3907242, 2.3950841
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8908887, 2.8860960
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1266274, 3.1266246
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0281725, 3.0224333
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4640312, 2.4699545

Time for backsubstitution: 13.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6766902, upper bound: 1.6759747
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6739708, upper bound: 1.6786688
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3907180, 2.3950856
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8908839, 2.8860979
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1266294, 3.1266184
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0281725, 3.0224361
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4640284, 2.4699559

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6781171, upper bound: 1.6745886
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6753521, upper bound: 1.6772419
time: 5.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3894563, 2.3884642
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8907981, 2.8857808
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1303730, 3.1264763
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3340445, 3.3407719
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0215702, 3.0178375
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4637389, 2.4651592

Time for backsubstitution: 13.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6881407, upper bound: 1.6839208
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6875403, upper bound: 1.6845000
time: 3.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3884954, 2.3894250
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8904376, 2.8861413
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1290216, 3.1278281
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3418455, 3.3329704
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0240469, 3.0153618
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4646239, 2.4642742

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6836466, upper bound: 1.6888711
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6837605, upper bound: 1.6887571
time: 4.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6887577, upper bound: 1.6837600
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6843821, upper bound: 1.6881356
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6883393, upper bound: 1.6882244
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6886773, upper bound: 1.6878862
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6738861, upper bound: 1.6821968
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6888727, upper bound: 1.6672289
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6829669, upper bound: 1.6880285
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6828822, upper bound: 1.6881132
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6766902, upper bound: 1.6759747
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6739708, upper bound: 1.6786688
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6781171, upper bound: 1.6745886
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6753521, upper bound: 1.6772419
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6881407, upper bound: 1.6839208
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6875403, upper bound: 1.6845000
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6836466, upper bound: 1.6888711
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 6, lower bound: -1.6837605, upper bound: 1.6887571

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3888707, 2.3993053
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8858466, 2.8958855
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1270247, 3.1414313
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3342052, 3.3421590
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0146980, 3.0329046
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4660978, 2.4690573

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6887472, upper bound: 1.6778380
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6828357, upper bound: 1.6837496
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3879104, 2.4002662
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8854861, 2.8962460
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1256733, 3.1427822
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3420072, 3.3343577
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0171747, 3.0304284
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4669828, 2.4681723

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6693754, upper bound: 1.6881163
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6843619, upper bound: 1.6731472
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3901386, 2.3992491
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8846130, 2.8984823
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1228948, 3.1385617
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3493190, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0199957, 3.0396237
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4659705, 2.4665675

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6769702, upper bound: 1.6783042
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6783975, upper bound: 1.6768774
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3907986, 2.3971364
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8856707, 2.8951435
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1236634, 3.1361165
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3489895
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0213709, 3.0352521
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4672036, 2.4626529

Time for backsubstitution: 13.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6886769, upper bound: 1.6872896
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6880973, upper bound: 1.6878871
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3909550, 2.3803065
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8878188, 2.8850474
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1253085, 3.1235452
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3338943
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0079432, 3.0318933
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4666786, 2.4483054

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6738858, upper bound: 1.6815960
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6734069, upper bound: 1.6821961
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3755250, 2.3917651
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8829770, 2.8886404
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1220183, 3.1259980
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3275719, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0128937, 3.0252452
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4431944, 2.4657681

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888582, upper bound: 1.6671300
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6887736, upper bound: 1.6672144
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3910203, 2.3898611
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8845425, 2.8903270
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1242566, 3.1281700
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3493476, 3.3469090
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0219984, 3.0188069
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4648290, 2.4644341

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6826801, upper bound: 1.6852887
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6802582, upper bound: 1.6877511
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3913527, 2.3857963
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8844109, 2.8903270
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1244483, 3.1258254
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3493834, 3.3464632
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0220346, 3.0183687
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4652371, 2.4594505

Time for backsubstitution: 13.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6825310, upper bound: 1.6881002
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6828688, upper bound: 1.6877626
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3839674, 2.3863769
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8988218, 2.8962827
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1188650, 3.1142015
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3489342
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0314493, 3.0266123
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4625440, 2.4690251

Time for backsubstitution: 13.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6766802, upper bound: 1.6700521
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6707689, upper bound: 1.6759639
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3820171, 2.3883274
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8988218, 2.8988218
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1142035, 3.1188626
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0323496, 3.0257120
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4631028, 2.4684668

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6739599, upper bound: 1.6727477
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6680483, upper bound: 1.6786590
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3839617, 2.3863783
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8988218, 2.8962846
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1188669, 3.1141953
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3489351
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0314512, 3.0266142
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4625411, 2.4690270

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6781071, upper bound: 1.6686662
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6721958, upper bound: 1.6745777
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3820114, 2.3883288
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8988218, 2.8988218
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1142054, 3.1188564
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495274, 3.3495297
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0323515, 3.0257149
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4630990, 2.4684687

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6753413, upper bound: 1.6713204
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6694296, upper bound: 1.6772317
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3792410, 2.3721290
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8505988, 2.8606606
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1204195, 3.1206460
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3427653, 3.3353767
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0283360, 3.0150733
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4327931, 2.4136775

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6881303, upper bound: 1.6779988
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6822187, upper bound: 1.6839102
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3731203, 2.3782492
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8656735, 2.8455858
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1245432, 3.1165223
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3286510, 3.3494916
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0188060, 3.0246029
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4122577, 2.4342134

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6874214, upper bound: 1.6844951
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6875353, upper bound: 1.6843810
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3879414, 2.3965025
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8900123, 2.8915901
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1282177, 3.1380844
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3430805, 3.3328736
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0233831, 3.0238166
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4664488, 2.4641314

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6836462, upper bound: 1.6882910
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6830458, upper bound: 1.6888705
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3884954, 2.3888710
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8904376, 2.8857160
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1290216, 3.1270242
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3417492, 3.3329704
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0240469, 3.0146980
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4644814, 2.4642742

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6687720, upper bound: 1.6887379
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6837403, upper bound: 1.6737502
time: 4.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6887472, upper bound: 1.6778380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6828357, upper bound: 1.6837496
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6693754, upper bound: 1.6881163
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6843619, upper bound: 1.6731472
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6769702, upper bound: 1.6783042
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6783975, upper bound: 1.6768774
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6886769, upper bound: 1.6872896
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6880973, upper bound: 1.6878871
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6738858, upper bound: 1.6815960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6734069, upper bound: 1.6821961
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6888582, upper bound: 1.6671300
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6887736, upper bound: 1.6672144
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6826801, upper bound: 1.6852887
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6802582, upper bound: 1.6877511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6825310, upper bound: 1.6881002
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6828688, upper bound: 1.6877626
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6766802, upper bound: 1.6700521
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6707689, upper bound: 1.6759639
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6739599, upper bound: 1.6727477
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6680483, upper bound: 1.6786590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6781071, upper bound: 1.6686662
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6721958, upper bound: 1.6745777
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6753413, upper bound: 1.6713204
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6694296, upper bound: 1.6772317
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6881303, upper bound: 1.6779988
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6822187, upper bound: 1.6839102
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6874214, upper bound: 1.6844951
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6875353, upper bound: 1.6843810
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6836462, upper bound: 1.6882910
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6830458, upper bound: 1.6888705
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6687720, upper bound: 1.6887379
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.99
Output dim: 6, lower bound: -1.6837403, upper bound: 1.6737502

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3845015, 2.4005699
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8863211, 2.8942003
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1270285, 3.1414123
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3301849, 3.3433263
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0055580, 3.0355554
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4615192, 2.4703882

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6883966, upper bound: 1.6778244
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6887340, upper bound: 1.6774869
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3888707, 2.3949361
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8841619, 2.8958855
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1270056, 3.1414313
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3342052, 3.3381388
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0146980, 3.0237651
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4660978, 2.4644780

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6678294, upper bound: 1.6837308
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6828155, upper bound: 1.6687612
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3918810, 2.3888075
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8867350, 2.8926530
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1265116, 3.1403298
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3481755, 3.3165646
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0122232, 3.0321255
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4730053, 2.4507091

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6691551, upper bound: 1.6853733
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6666261, upper bound: 1.6878381
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.3764515, 2.4002662
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8818932, 2.8962460
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1232204, 3.1427822
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3242135, 3.3343577
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.0171747, 3.0254774
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.4495201, 2.4681723

Time for backsubstitution: 12.68 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.363316535949707
rel_dist={6: [-1.6889095697923429, 1.6889090371909665]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5621000, upper bound: 1.5587157
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5587151, upper bound: 1.5621007
time: 4.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 6, lower bound: -1.5621000, upper bound: 1.5587157
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 6, lower bound: -1.5587151, upper bound: 1.5621007

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2661500, 2.2654023
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7239013, 2.7236204
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9880495, 2.9869986
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1644058, 3.1704736
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3448162, 2.3443217
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8416777, 2.8436027
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3396549, 2.3403430

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5505338, upper bound: 1.5485349
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5519205, upper bound: 1.5471482
time: 4.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2654023, 2.2661495
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7236209, 2.7239013
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9869986, 2.9880495
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1704731, 3.1644058
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3443217, 2.3448162
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8436041, 2.8416772
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3403435, 2.3396549

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5587148, upper bound: 1.5618164
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5584310, upper bound: 1.5621000
time: 4.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.75 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.75
Output dim: 6, lower bound: -1.5505338, upper bound: 1.5485349
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.75
Output dim: 6, lower bound: -1.5519205, upper bound: 1.5471482
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.75
Output dim: 6, lower bound: -1.5587148, upper bound: 1.5618164
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.75
Output dim: 6, lower bound: -1.5584310, upper bound: 1.5621000

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2661524, 2.2654004
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7239032, 2.7236195
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9880486, 2.9870024
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1644092, 3.1704721
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3448148, 2.3443241
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8416767, 2.8436007
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3396559, 2.3403416

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5505335, upper bound: 1.5482879
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5502768, upper bound: 1.5485349
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2661476, 2.2654023
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7239003, 2.7236204
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9880495, 2.9869971
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1644044, 3.1704736
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3448162, 2.3443203
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8416777, 2.8436027
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3396535, 2.3403430

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5519203, upper bound: 1.5468911
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5516733, upper bound: 1.5471482
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2538261, 2.2498133
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6834240, 2.6954293
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9770451, 2.9813042
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1760573, 3.1590109
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3459592, 2.3481028
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8482504, 2.8389120
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3048339, 2.2881730

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5436799, upper bound: 1.5617968
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5586947, upper bound: 1.5467881
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2490664, 2.2545738
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6951485, 2.6837044
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9802532, 2.9780970
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1650786, 3.1699891
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3476086, 2.3464534
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8408384, 2.8463244
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2888618, 2.3041453

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5584183, upper bound: 1.5620031
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583340, upper bound: 1.5620872
time: 4.23 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.60 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5505335, upper bound: 1.5482879
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5502768, upper bound: 1.5485349
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5519203, upper bound: 1.5468911
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5516733, upper bound: 1.5471482
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5436799, upper bound: 1.5617968
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5586947, upper bound: 1.5467881
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5584183, upper bound: 1.5620031
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.60
Output dim: 6, lower bound: -1.5583340, upper bound: 1.5620872

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2545767, 2.2490647
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6837063, 2.6951475
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780951, 2.9802566
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1699934, 3.1650772
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3464522, 2.3476107
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8463249, 2.8408365
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3041468, 2.2888601

Time for backsubstitution: 12.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5505223, upper bound: 1.5425898
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5448367, upper bound: 1.5482767
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2498169, 2.2538249
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6954327, 2.6834230
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9813023, 2.9770489
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1590147, 3.1760554
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3481016, 2.3459613
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8389130, 2.8482485
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2881742, 2.3048325

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5499357, upper bound: 1.5485215
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5502635, upper bound: 1.5481948
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2545719, 2.2490661
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6837044, 2.6951485
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780970, 2.9802513
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1699877, 3.1650786
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3464537, 2.3476069
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8463240, 2.8408384
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3041444, 2.2888613

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5470645, upper bound: 1.5435889
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5518913, upper bound: 1.5433963
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2498121, 2.2538264
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6954288, 2.6834240
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9813042, 2.9770441
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1590090, 3.1760573
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3481030, 2.3459575
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8389120, 2.8482509
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2881722, 2.3048337

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5516606, upper bound: 1.5470511
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5515764, upper bound: 1.5471354
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2543688, 2.2383554
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6835980, 2.6918368
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9771504, 2.9788504
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1769018, 3.1412191
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3354344, 2.3485987
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8433018, 2.8391337
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3056440, 2.2707107

Time for backsubstitution: 13.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5436781, upper bound: 1.5612445
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5431414, upper bound: 1.5617950
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2423677, 2.2498133
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6798310, 2.6954293
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9745917, 2.9813042
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1582651, 3.1590109
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3459592, 2.3375781
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8482504, 2.8339629
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2873716, 2.2881730

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5438657, upper bound: 1.5467873
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5485057, upper bound: 1.5467811
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2487340, 2.2574034
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6952429, 2.6837010
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9800606, 2.9797277
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1650424, 3.1702995
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3495493, 2.3462265
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8408022, 2.8466291
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2884545, 2.3076146

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583407, upper bound: 1.5596861
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560848, upper bound: 1.5619251
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2490664, 2.2542419
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6951456, 2.6837044
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9802532, 2.9779043
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1650786, 3.1699533
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3473811, 2.3464534
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8408384, 2.8462887
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2888618, 2.3037386

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583321, upper bound: 1.5615425
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5577813, upper bound: 1.5620849
time: 4.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5505223, upper bound: 1.5425898
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5448367, upper bound: 1.5482767
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5499357, upper bound: 1.5485215
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5502635, upper bound: 1.5481948
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5470645, upper bound: 1.5435889
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5518913, upper bound: 1.5433963
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5516606, upper bound: 1.5470511
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5515764, upper bound: 1.5471354
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5436781, upper bound: 1.5612445
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5431414, upper bound: 1.5617950
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5438657, upper bound: 1.5467873
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5485057, upper bound: 1.5467811
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5583407, upper bound: 1.5596861
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5560848, upper bound: 1.5619251
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5583321, upper bound: 1.5615425
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.54
Output dim: 6, lower bound: -1.5577813, upper bound: 1.5620849

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2502074, 2.2490795
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6837010, 2.6934614
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780941, 2.9802375
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1659727, 3.1650920
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3461881, 2.3476105
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8371849, 2.8408666
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2995677, 2.2888772

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5501818, upper bound: 1.5425758
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5505083, upper bound: 1.5422491
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2545767, 2.2446954
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6820207, 2.6951475
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780760, 2.9802566
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1699934, 3.1610575
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3464522, 2.3473463
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8463249, 2.8316960
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3041468, 2.2842805

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5448141, upper bound: 1.5458231
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5424247, upper bound: 1.5482761
time: 4.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2491560, 2.2548077
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6943741, 2.6849623
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9805336, 2.9781818
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1576176, 3.1781168
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3479395, 2.3461950
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8375359, 2.8502722
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2869415, 2.3066444

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5499338, upper bound: 1.5479586
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5493767, upper bound: 1.5485195
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2498169, 2.2531645
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6954327, 2.6823654
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9813023, 2.9762802
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1590147, 3.1746593
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3481016, 2.3457992
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8389130, 2.8468723
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2881742, 2.3035998

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5502616, upper bound: 1.5476317
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5497045, upper bound: 1.5481928
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2551169, 2.2376080
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6838784, 2.6915560
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9782023, 2.9777970
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1708341, 3.1472869
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3359289, 2.3481026
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8413754, 2.8410587
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3049545, 2.2713990

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5470518, upper bound: 1.5434917
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5469674, upper bound: 1.5435762
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2431159, 2.2490661
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6801114, 2.6951485
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9756427, 2.9802513
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1521974, 3.1650786
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3464537, 2.3370824
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8463240, 2.8358879
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2866817, 2.2888613

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5518908, upper bound: 1.5410770
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5494361, upper bound: 1.5433246
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2494798, 2.2566562
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6955223, 2.6834207
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9811115, 2.9786754
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1589737, 3.1763673
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3500438, 2.3457305
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8388767, 2.8485546
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2877650, 2.3083031

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5516587, upper bound: 1.5464927
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5510972, upper bound: 1.5470494
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2498121, 2.2534945
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6954250, 2.6834240
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9813042, 2.9768515
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1590090, 3.1760211
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3478756, 2.3459575
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8389120, 2.8482141
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2881722, 2.3044269

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5512363, upper bound: 1.5471221
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5515632, upper bound: 1.5467941
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2538009, 2.2375526
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6960163, 2.7078772
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9754534, 2.9789829
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1790762, 3.1442003
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3370419, 2.3506727
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8267875, 2.8274221
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3038559, 2.2681961

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 5817

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5435605, upper bound: 1.5612391
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5436726, upper bound: 1.5611270
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2535663, 2.2377868
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6996374, 2.7042556
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9772835, 2.9771528
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1798906, 3.1433926
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3375125, 2.3502064
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8315892, 2.8226204
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3031297, 2.2689290

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5431287, upper bound: 1.5616978
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5430443, upper bound: 1.5617820
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2423730, 2.2498119
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6798348, 2.6954284
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9745898, 2.9813075
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1582704, 3.1590095
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3459578, 2.3375807
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8482504, 2.8339596
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2873726, 2.2881718

Time for backsubstitution: 12.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5438546, upper bound: 1.5410867
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5381650, upper bound: 1.5467762
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2423682, 2.2498133
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6798310, 2.6954293
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9745917, 2.9813027
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1582646, 3.1590109
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3459592, 2.3375769
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8482504, 2.8339615
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2873697, 2.2881730

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5485052, upper bound: 1.5444622
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5460521, upper bound: 1.5467096
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2419777, 2.2491298
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7087512, 2.6938858
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9712601, 2.9673018
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1650715, 3.1693478
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3491545, 2.3456686
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8442817, 2.8508081
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2869668, 2.3065612

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5579974, upper bound: 1.5596729
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583273, upper bound: 1.5593417
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2404613, 2.2506471
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7054276, 2.6972089
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9676342, 2.9709272
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1640911, 3.1703281
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3489914, 2.3458316
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8449807, 2.8501081
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2874012, 2.3061271

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5410711, upper bound: 1.5619063
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5560638, upper bound: 1.5468963
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2484975, 2.2534392
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7075629, 2.6997452
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9785547, 2.9780359
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1672530, 3.1729345
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3489885, 2.3485308
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8243251, 2.8345780
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2870803, 2.3012235

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5433038, upper bound: 1.5615222
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5583124, upper bound: 1.5465142
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2482634, 2.2536733
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7111850, 2.6961231
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9803848, 2.9762058
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1680617, 3.1721268
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3494587, 2.3480606
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8291278, 2.8297763
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2863469, 2.3019567

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5577698, upper bound: 1.5563844
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5520820, upper bound: 1.5620736
time: 4.47 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5501818, upper bound: 1.5425758
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5505083, upper bound: 1.5422491
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5448141, upper bound: 1.5458231
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5424247, upper bound: 1.5482761
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5499338, upper bound: 1.5479586
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5493767, upper bound: 1.5485195
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5502616, upper bound: 1.5476317
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5497045, upper bound: 1.5481928
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5470518, upper bound: 1.5434917
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5469674, upper bound: 1.5435762
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5518908, upper bound: 1.5410770
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5494361, upper bound: 1.5433246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5516587, upper bound: 1.5464927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5510972, upper bound: 1.5470494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5512363, upper bound: 1.5471221
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5515632, upper bound: 1.5467941
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5435605, upper bound: 1.5612391
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5436726, upper bound: 1.5611270
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5431287, upper bound: 1.5616978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5430443, upper bound: 1.5617820
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5438546, upper bound: 1.5410867
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5381650, upper bound: 1.5467762
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5485052, upper bound: 1.5444622
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5460521, upper bound: 1.5467096
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5579974, upper bound: 1.5596729
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5583273, upper bound: 1.5593417
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5410711, upper bound: 1.5619063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5560638, upper bound: 1.5468963
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5433038, upper bound: 1.5615222
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5583124, upper bound: 1.5465142
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5577698, upper bound: 1.5563844
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.64
Output dim: 6, lower bound: -1.5520820, upper bound: 1.5620736

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2495489, 2.2500634
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6826425, 2.6950006
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9773254, 2.9813700
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1645780, 3.1671572
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3460269, 2.3478453
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8358078, 2.8428936
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2983360, 2.2906942

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5501621, upper bound: 1.5401259
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5477636, upper bound: 1.5425750
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2502074, 2.2484202
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6837010, 2.6924043
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9780941, 2.9794683
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1659727, 3.1636963
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3461881, 2.3474495
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8371849, 2.8394909
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2995677, 2.2876463

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5503907, upper bound: 1.5422436
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5505029, upper bound: 1.5421314
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2478204, 2.2364218
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6955290, 2.7053328
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9692745, 2.9678292
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1700211, 3.1601057
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3460579, 2.3467891
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8498034, 2.8358736
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3026590, 2.2832272

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5448122, upper bound: 1.5452597
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5442557, upper bound: 1.5458214
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2463031, 2.2379386
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6922064, 2.7086568
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9656487, 2.9714546
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1690416, 3.1610861
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3458948, 2.3469522
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8505034, 2.8351736
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.3030930, 2.2827930

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5424228, upper bound: 1.5477137
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5418672, upper bound: 1.5482741
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2485867, 2.2540045
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.7067914, 2.7010012
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9788351, 2.9783134
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.1597924, 3.1810994
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3495474, 2.3482735
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.8210244, 2.8385615
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2851601, 2.3041294

Time for backsubstitution: 12.84 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.3460235595703125
rel_dist={6: [-1.5621134080175434, 1.562113956458468]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4558
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4558

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944187, upper bound: 1.4915169
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4915170, upper bound: 1.4944191
time: 4.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.23
Output dim: 6, lower bound: -1.4944187, upper bound: 1.4915169
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.23
Output dim: 6, lower bound: -1.4915170, upper bound: 1.4944191

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2040944, 2.2034543
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6489892, 2.6487489
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9172163, 2.9163151
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0811920, 3.0863934
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3083682, 2.3079443
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7465611, 2.7482114
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2763557, 2.2769454

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943674, upper bound: 1.4896680
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925401, upper bound: 1.4914517
time: 4.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.2034545, 2.2040949
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6487489, 2.6489897
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9163151, 2.9172158
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0863934, 3.0811920
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3079443, 2.3083677
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7482119, 2.7465606
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2769456, 2.2763553

Time for backsubstitution: 13.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6196

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4915167, upper bound: 1.4942205
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4913174, upper bound: 1.4944185
time: 4.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.37 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 6, lower bound: -1.4943674, upper bound: 1.4896680
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 6, lower bound: -1.4925401, upper bound: 1.4914517
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 6, lower bound: -1.4915167, upper bound: 1.4942205
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.37
Output dim: 6, lower bound: -1.4913174, upper bound: 1.4944185

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1973367, 2.1953964
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6620216, 2.6589332
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9078984, 2.9038897
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0810814, 3.0854421
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3079500, 2.3073864
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7501402, 2.7523899
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2748680, 2.2758303

Time for backsubstitution: 13.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940496, upper bound: 1.4896544
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943539, upper bound: 1.4893505
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1960363, 2.1966968
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6591740, 2.6617823
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9047904, 2.9069967
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0802412, 3.0862823
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3078103, 2.3075261
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7507391, 2.7517900
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2752404, 2.2754581

Time for backsubstitution: 13.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925279, upper bound: 1.4913553
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4924438, upper bound: 1.4914395
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1911983, 2.1877587
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6085529, 2.6188426
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9063616, 2.9100122
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0904088, 3.0757976
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3095818, 2.3114192
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7517996, 2.7437959
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2391543, 2.2248735

Time for backsubstitution: 13.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4915149, upper bound: 1.4937384
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4910317, upper bound: 1.4942188
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1871176, 2.1918390
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6186028, 2.6087928
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9091110, 2.9072628
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0809979, 3.0852075
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3109956, 2.3100054
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7454462, 2.7501488
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2254643, 2.2385643

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4584

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4913155, upper bound: 1.4939353
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4908345, upper bound: 1.4944165
time: 5.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4940496, upper bound: 1.4896544
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4943539, upper bound: 1.4893505
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4925279, upper bound: 1.4913553
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4924438, upper bound: 1.4914395
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4915149, upper bound: 1.4937384
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4910317, upper bound: 1.4942188
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4913155, upper bound: 1.4939353
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.40
Output dim: 6, lower bound: -1.4908345, upper bound: 1.4944165

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1966772, 2.1961443
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6609640, 2.6601009
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9071307, 2.9047518
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0796843, 3.0870094
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3077888, 2.3075643
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7487640, 2.7539291
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2736349, 2.2772067

Time for backsubstitution: 13.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940420, upper bound: 1.4847546
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891300, upper bound: 1.4896454
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1973367, 2.1947360
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6620216, 2.6578751
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9078984, 2.9031219
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0810814, 3.0840454
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3079500, 2.3072252
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7501402, 2.7510147
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2748680, 2.2745972

Time for backsubstitution: 13.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943416, upper bound: 1.4892546
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4942575, upper bound: 1.4893384
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1957049, 2.1990747
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6592607, 2.6617823
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9045997, 2.9083686
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0802050, 3.0865431
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3094420, 2.3072996
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7507029, 2.7520466
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2748322, 2.2783725

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4789914, upper bound: 1.4913443
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925165, upper bound: 1.4779020
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1960363, 2.1963649
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6591749, 2.6617823
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9047904, 2.9068050
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0802412, 3.0862465
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3075838, 2.3075261
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7507391, 2.7517538
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2752404, 2.2750502

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4924355, upper bound: 1.4865193
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4875248, upper bound: 1.4914312
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1905961, 2.1869557
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6209712, 2.6343656
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9046631, 2.9098816
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0925832, 3.0786648
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3111887, 2.3134294
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7352872, 2.7313995
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2372684, 2.2223594

Time for backsubstitution: 13.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4914754, upper bound: 1.4937361
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4915126, upper bound: 1.4936989
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1903954, 2.1871562
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6240745, 2.6312613
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9062319, 2.9083128
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0932755, 3.0779724
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3115916, 2.3130260
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7394032, 2.7272835
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2366400, 2.2229877

Time for backsubstitution: 13.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4775510, upper bound: 1.4942076
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4910192, upper bound: 1.4807521
time: 4.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1865158, 2.1910360
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6310210, 2.6243157
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9074125, 2.9071326
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0831733, 3.0880747
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3126020, 2.3120155
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7289338, 2.7377520
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2235780, 2.2360501

Time for backsubstitution: 13.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4910023, upper bound: 1.4939214
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4913020, upper bound: 1.4936177
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1863151, 2.1912365
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6341243, 2.6212115
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9089813, 2.9055638
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0838656, 3.0873823
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3130054, 2.3116126
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7330499, 2.7336359
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2229495, 2.2366784

Time for backsubstitution: 13.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4908225, upper bound: 1.4943202
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4907384, upper bound: 1.4944042
time: 4.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4940420, upper bound: 1.4847546
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4891300, upper bound: 1.4896454
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4943416, upper bound: 1.4892546
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4942575, upper bound: 1.4893384
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4789914, upper bound: 1.4913443
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4925165, upper bound: 1.4779020
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4924355, upper bound: 1.4865193
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4875248, upper bound: 1.4914312
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4914754, upper bound: 1.4937361
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4915126, upper bound: 1.4936989
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4775510, upper bound: 1.4942076
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4910192, upper bound: 1.4807521
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4910023, upper bound: 1.4939214
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4913020, upper bound: 1.4936177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4908225, upper bound: 1.4943202
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.63
Output dim: 6, lower bound: -1.4907384, upper bound: 1.4944042

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1923079, 2.1955335
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6607175, 2.6584153
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9071264, 2.9047322
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0756655, 3.0864515
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3075252, 2.3075266
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7396240, 2.7526522
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2690578, 2.2765725

Time for backsubstitution: 13.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4839789, upper bound: 1.4758723
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4853400, upper bound: 1.4745340
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1960659, 2.1917760
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6592774, 2.6598554
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9071112, 2.9047484
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0791245, 3.0829906
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3077512, 2.3073001
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7474842, 2.7447891
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2729983, 2.2726295

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891174, upper bound: 1.4895496
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4890336, upper bound: 1.4896336
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1970057, 2.1971140
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6621103, 2.6578755
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9077067, 2.9044933
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0810452, 3.0843062
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3095818, 2.3069987
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7501030, 2.7512717
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2744603, 2.2775118

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943330, upper bound: 1.4843519
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4894211, upper bound: 1.4892469
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1973367, 2.1944039
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6620226, 2.6578751
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9078984, 2.9029303
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0810814, 3.0840092
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3077235, 2.3072252
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7501402, 2.7509799
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2748680, 2.2741892

Time for backsubstitution: 13.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4841948, upper bound: 1.4804510
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4855564, upper bound: 1.4791127
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1945329, 2.1876163
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6588964, 2.6581893
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9043384, 2.9059148
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0783877, 3.0687513
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.2989173, 2.3062210
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7457523, 2.7515278
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2730322, 2.2609098

Time for backsubstitution: 13.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4654

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4786692, upper bound: 1.4913306
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4789779, upper bound: 1.4910265
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1842461, 2.1979027
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6556692, 2.6614175
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9021449, 2.9081082
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0624127, 3.0847259
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3083634, 2.2967749
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7501841, 2.7470951
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2573700, 2.2765725

Time for backsubstitution: 13.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4791534, upper bound: 1.4779014
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4836910, upper bound: 1.4778966
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1916676, 2.1957529
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6589270, 2.6600962
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9047861, 2.9067860
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0762205, 3.0856838
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3073201, 2.3074889
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7415972, 2.7504740
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2706609, 2.2744117

Time for backsubstitution: 13.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4822932, upper bound: 1.4778290
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4836315, upper bound: 1.4764640
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1954250, 2.1919954
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6574879, 2.6615357
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9047709, 2.9068012
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0796785, 3.0822253
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3075466, 2.3072629
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7494583, 2.7426138
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2746015, 2.2704718

Time for backsubstitution: 13.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4773960, upper bound: 1.4827297
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4787352, upper bound: 1.4813682
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1900420, 2.1914895
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6205440, 2.6378546
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9038591, 2.9164505
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0933733, 3.0785675
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3120813, 2.3133197
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7346234, 2.7368145
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2384367, 2.2222164

Time for backsubstitution: 13.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4779911, upper bound: 1.4937240
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4914640, upper bound: 1.4802744
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1905961, 2.1864021
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6209712, 2.6339383
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9046631, 2.9090776
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0924854, 3.0786648
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3110790, 2.3134294
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7352872, 2.7307358
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2371254, 2.2223594

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4915043, upper bound: 1.4887979
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4865934, upper bound: 1.4936905
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1892238, 2.1756983
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6237106, 2.6276689
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9059734, 2.9058604
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0914640, 3.0601797
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3010674, 2.3119483
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7344527, 2.7267647
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2348399, 2.2055249

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4775453, upper bound: 1.4855060
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4775508, upper bound: 1.4809122
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1789374, 2.1859848
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6204834, 2.6308966
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9037790, 2.9080544
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0754824, 3.0761542
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3105102, 2.3025022
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7388844, 2.7223330
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2191772, 2.2211814

Time for backsubstitution: 13.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4909528, upper bound: 1.4788244
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4891774, upper bound: 1.4807362
time: 4.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1858549, 2.1917834
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6299615, 2.6254826
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9066439, 2.9079938
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0817771, 3.0896420
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3124413, 2.3121939
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7275586, 2.7392907
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2223444, 2.2374263

Time for backsubstitution: 13.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 5817
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4775353, upper bound: 1.4939088
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4909909, upper bound: 1.4804521
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1865158, 2.1903751
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6310210, 2.6232567
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9074125, 2.9063640
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0831733, 3.0866785
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3126020, 2.3118548
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7289338, 2.7363772
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2235780, 2.2348166

Time for backsubstitution: 13.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5817

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912932, upper bound: 1.4886946
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4864005, upper bound: 1.4936102
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1859832, 2.1936147
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6342044, 2.6212072
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9087896, 2.9069357
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0838284, 3.0876422
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3146367, 2.3113852
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7330136, 2.7338929
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2225418, 2.2395930

Time for backsubstitution: 13.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 4654
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4908141, upper bound: 1.4894010
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4859188, upper bound: 1.4943120
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1863151, 2.1909049
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6341214, 2.6212115
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9089813, 2.9053721
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0838656, 3.0873451
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3127785, 2.3116126
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7330499, 2.7336011
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2229495, 2.2362707

Time for backsubstitution: 13.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6221
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4772768, upper bound: 1.4943929
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4907260, upper bound: 1.4809335
time: 5.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4839789, upper bound: 1.4758723
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4853400, upper bound: 1.4745340
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4891174, upper bound: 1.4895496
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4890336, upper bound: 1.4896336
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4943330, upper bound: 1.4843519
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4894211, upper bound: 1.4892469
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4841948, upper bound: 1.4804510
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4855564, upper bound: 1.4791127
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4786692, upper bound: 1.4913306
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4789779, upper bound: 1.4910265
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4791534, upper bound: 1.4779014
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4836910, upper bound: 1.4778966
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4822932, upper bound: 1.4778290
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4836315, upper bound: 1.4764640
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4773960, upper bound: 1.4827297
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4787352, upper bound: 1.4813682
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4779911, upper bound: 1.4937240
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4914640, upper bound: 1.4802744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4915043, upper bound: 1.4887979
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4865934, upper bound: 1.4936905
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4775453, upper bound: 1.4855060
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4775508, upper bound: 1.4809122
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4909528, upper bound: 1.4788244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4891774, upper bound: 1.4807362
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4775353, upper bound: 1.4939088
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4909909, upper bound: 1.4804521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4912932, upper bound: 1.4886946
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4864005, upper bound: 1.4936102
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4908141, upper bound: 1.4894010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4859188, upper bound: 1.4943120
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4772768, upper bound: 1.4943929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 6, lower bound: -1.4907260, upper bound: 1.4809335

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1957340, 2.1941521
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6593657, 2.6598554
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9069185, 2.9061193
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0790882, 3.0832515
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3093829, 2.3070736
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7474499, 2.7450457
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2725902, 2.2755418

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4790615, upper bound: 1.4806670
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4804261, upper bound: 1.4793288
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1960659, 2.1914439
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6592779, 2.6598554
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9071112, 2.9045563
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0791245, 3.0829549
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3075247, 2.3073001
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7474842, 2.7447538
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2729983, 2.2722218

Time for backsubstitution: 13.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 6196
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4755804, upper bound: 1.4896223
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4890223, upper bound: 1.4760106
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.1926355, 2.1965015
1: -10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.6618633, 2.6561894
2: -5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799
3: -12.1876364, -8.9962378, -12.1876364, -8.9962378, -2.9077024, 2.9044733
4: -8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.0770245, 3.0837464
5: -0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036
6: 5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3093181, 2.3069606
7: -18.8783417, -15.4095268, -18.8783417, -15.4095268, -2.7409620, 2.7499924
8: -1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653
9: -8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.2698817, 2.2768745

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 481
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4584
type: RSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 481

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4808798, upper bound: 1.4843406
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943218, upper bound: 1.4707274
time: 4.36 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.48
Output dim: 6, lower bound: -1.4790615, upper bound: 1.4806670
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.48
Output dim: 6, lower bound: -1.4804261, upper bound: 1.4793288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.48
Output dim: 6, lower bound: -1.4755804, upper bound: 1.4896223
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.48
Output dim: 6, lower bound: -1.4890223, upper bound: 1.4760106
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.48
Output dim: 6, lower bound: -1.4808798, upper bound: 1.4843406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.48
Output dim: 6, lower bound: -1.4943218, upper bound: 1.4707274
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4894211, upper bound: 1.4892469
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4786692, upper bound: 1.4913306
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4789779, upper bound: 1.4910265
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4779911, upper bound: 1.4937240
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4914640, upper bound: 1.4802744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4915043, upper bound: 1.4887979
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4865934, upper bound: 1.4936905
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4909528, upper bound: 1.4788244
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4891774, upper bound: 1.4807362
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4775353, upper bound: 1.4939088
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4909909, upper bound: 1.4804521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4912932, upper bound: 1.4886946
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4864005, upper bound: 1.4936102
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4908141, upper bound: 1.4894010
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4859188, upper bound: 1.4943120
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4772768, upper bound: 1.4943929
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.48
Output dim: 6, lower bound: -1.4907260, upper bound: 1.4809335
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.309645652770996
rel_dist={6: [-1.4944306407619283, 1.4944305738519086]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2419.71 seconds
