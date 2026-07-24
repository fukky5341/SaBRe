## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.44380584438
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.6581917, 1.6581917)
1: (-2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.9685266, 1.9685266)
2: (-4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.8327081, 1.8327081)
3: (-12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.6218438, 2.6218441)
4: (-6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.7623563, 1.7623563)
5: (-2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.8194118, 1.8194118)
6: (2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445)
7: (-10.2777958, -8.1927624, -10.2777958, -8.1927624, -2.0850334, 2.0850334)
8: (-1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.4304194, 2.4304194)
9: (-8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.5183010, 1.5183010)

## BASE Result
execution time: IAR + LP analysis = 13.08 + 32.00 = 45.07 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.93 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.568244457244873
rel_dist={6: [-0.8170414609587353, 0.8170398270121066]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.525653600692749
rel_dist={6: [-0.6052081522674859, 0.6052060697814263]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.4518237113952637
rel_dist={6: [-0.44793373691976335, 0.4479324567017895]}

## Binary Search Result
Binary search time: 140.98 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 3413.94 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8037509, upper bound: 0.8037495
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8037509, upper bound: 0.8066939
time: 5.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.53
Output dim: 6, lower bound: -0.8037509, upper bound: 0.8037495
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.53
Output dim: 6, lower bound: -0.8037509, upper bound: 0.8066939

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3543122, 1.3543100
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6292121, 1.6292053
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6463027, 1.6462998
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1670089, 2.1670144
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5472264, 1.5472159
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5869460, 1.5869522
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8604894, 1.8604956
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1077538, 2.1077533
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4621453, 1.4621499

Time for backsubstitution: 13.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8009685, upper bound: 0.8036939
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8066785, upper bound: 0.8009662
time: 4.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3543127, 1.3543119
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6292055, 1.6292095
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6462998, 1.6463012
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1670117, 2.1670091
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5472159, 1.5472209
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5869489, 1.5869460
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8604941, 1.8604894
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1077533, 2.1077538
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4621496, 1.4621456

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7992077, upper bound: 0.7983692
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983711, upper bound: 0.8066278
time: 4.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.19
Output dim: 6, lower bound: -0.8009685, upper bound: 0.8036939
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.19
Output dim: 6, lower bound: -0.8066785, upper bound: 0.8009662
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.19
Output dim: 6, lower bound: -0.7992077, upper bound: 0.7983692
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.19
Output dim: 6, lower bound: -0.7983711, upper bound: 0.8066278

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3601148, 1.3496912
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6275623, 1.6312684
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6427398, 1.6507971
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1630087, 2.1720598
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5525301, 1.5430038
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5930007, 1.5821438
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8553972, 1.8669183
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1110544, 2.1051221
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4674621, 1.4579160

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8009626, upper bound: 0.8036877
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8009626, upper bound: 0.8036877
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3496931, 1.3543100
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6292121, 1.6275557
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6463027, 1.6427372
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1670089, 2.1630139
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5430148, 1.5472159
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5821373, 1.5869522
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8604894, 1.8554034
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1051221, 2.1077533
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4579120, 1.4621499

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8066152, upper bound: 0.7956322
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956339, upper bound: 0.7962839
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3517022, 1.3535080
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6277671, 1.6245263
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6368055, 1.6433908
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443429, 2.1600711
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5442338, 1.5374825
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5800805, 1.5645015
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8307476, 1.8513820
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1075501, 2.1076903
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4560018, 1.4602582

Time for backsubstitution: 13.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7962841, upper bound: 0.7983490
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7990920, upper bound: 0.7956322
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3535085, 1.3517013
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6245217, 1.6277717
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6433897, 1.6368068
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600747, 2.1443393
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5374770, 1.5442390
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5645046, 1.5800774
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8513865, 1.8307428
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1076899, 2.1075506
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4602618, 1.4559982

Time for backsubstitution: 13.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983651, upper bound: 0.8066224
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983651, upper bound: 0.8066216
time: 4.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.8009626, upper bound: 0.8036877
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.8009626, upper bound: 0.8036877
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.8066152, upper bound: 0.7956322
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.7956339, upper bound: 0.7962839
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.7962841, upper bound: 0.7983490
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.7990920, upper bound: 0.7956322
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.7983651, upper bound: 0.8066224
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.09
Output dim: 6, lower bound: -0.7983651, upper bound: 0.8066216

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3597541, 1.3509483
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6296575, 1.6306759
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6429856, 1.6507301
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1629705, 2.1721952
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5531881, 1.5428193
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5943887, 1.5817492
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8573093, 1.8663807
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1121945, 2.1048076
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4700518, 1.4571884

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3601148, 1.3493304
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6269701, 1.6312684
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6426728, 1.6507971
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1630087, 2.1720219
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5523460, 1.5430038
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5926063, 1.5821438
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8548598, 1.8669183
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1107402, 2.1051221
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4667344, 1.4579160

Time for backsubstitution: 13.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3470826, 1.3535097
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6277771, 1.6228722
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6368084, 1.6398268
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443396, 2.1560788
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5400333, 1.5374773
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5752676, 1.5645077
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8307428, 1.8462992
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1049190, 2.1076896
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4517646, 1.4602675

Time for backsubstitution: 13.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3488889, 1.3516996
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6245284, 1.6261177
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6433868, 1.6332428
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600714, 2.1403449
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5332766, 1.5442338
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5596917, 1.5800710
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8513823, 1.8256557
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1050582, 2.1075501
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4560246, 1.4560025

Time for backsubstitution: 13.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983427, upper bound: 0.7962784
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983427, upper bound: 0.7962784
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3575048, 1.3488889
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6261177, 1.6265893
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6332426, 1.6478894
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403422, 2.1651177
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5495391, 1.5332704
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5861280, 1.5596919
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8256545, 1.8577995
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1108508, 2.1050580
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4613180, 1.4560242

Time for backsubstitution: 13.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3470831, 1.3535080
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6277671, 1.6228766
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6368055, 1.6398280
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443429, 2.1560714
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5400219, 1.5374825
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5752709, 1.5645015
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8307476, 1.8462889
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1049190, 2.1076903
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4517674, 1.4602582

Time for backsubstitution: 13.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3531480, 1.3529587
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6266177, 1.6271793
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6436355, 1.6367397
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600366, 2.1444745
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5381355, 1.5440540
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5658927, 1.5796831
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8532987, 1.8302054
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1088305, 2.1072371
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4628518, 1.4552705

Time for backsubstitution: 13.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066097
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983427, upper bound: 0.8038854
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3535085, 1.3513405
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6239297, 1.6277717
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6433227, 1.6368068
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600747, 2.1443012
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5372925, 1.5442390
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5641103, 1.5800774
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8508492, 1.8307428
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1073761, 2.1075506
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4595344, 1.4559982

Time for backsubstitution: 13.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066104
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983427, upper bound: 0.8038854
time: 4.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7983427, upper bound: 0.7962784
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7983427, upper bound: 0.7962784
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066097
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7983427, upper bound: 0.8038854
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066104
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.58
Output dim: 6, lower bound: -0.7983427, upper bound: 0.8038854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3571434, 1.3501480
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6282237, 1.6259928
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6334908, 1.6478212
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403012, 2.1652601
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5502090, 1.5330811
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5875127, 1.5593038
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8275619, 1.8572721
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1119914, 2.1047440
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4639044, 1.4553058

Time for backsubstitution: 13.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2455

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2588

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7914335, upper bound: 0.7982893
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8038330, upper bound: 0.7858911
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3589468, 1.3483379
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6249745, 1.6292359
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6400702, 1.6412358
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1560330, 2.1495256
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5434504, 1.5398376
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5719435, 1.5748670
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8482008, 1.8366332
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1121287, 2.1046045
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4681611, 1.4510410

Time for backsubstitution: 13.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7817588, upper bound: 0.7712611
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7680794, upper bound: 0.7849572
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3575044, 1.3485301
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6255352, 1.6265849
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6331789, 1.6478882
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403394, 2.1650867
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5493660, 1.5332656
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5857303, 1.5596981
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8251119, 1.8578098
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1105371, 2.1050584
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4605870, 1.4560337

Time for backsubstitution: 13.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7964890, upper bound: 0.7834296
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7890189, upper bound: 0.7907789
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3593073, 1.3467200
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6222866, 1.6298280
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6397574, 1.6413028
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1560712, 2.1493523
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5426073, 1.5400221
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5701611, 1.5752614
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8457513, 1.8371706
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1106739, 2.1049190
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4648438, 1.4517686

Time for backsubstitution: 13.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1843

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7306726, upper bound: 0.7345219
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7306704, upper bound: 0.7345225
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3467216, 1.3547671
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6298730, 1.6222801
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6370542, 1.6397598
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443009, 2.1562138
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5406914, 1.5372927
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5766556, 1.5641134
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8326550, 1.8457618
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1060596, 2.1073759
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4543538, 1.4595399

Time for backsubstitution: 13.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1696

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7970597, upper bound: 0.7945283
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8053996, upper bound: 0.7861005
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3470826, 1.3531489
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6271851, 1.6228722
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6367414, 1.6398268
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443396, 2.1560404
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5398483, 1.5374773
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5748732, 1.5645077
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8302050, 1.8462992
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1046047, 2.1076896
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4510365, 1.4602675

Time for backsubstitution: 13.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 710

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7896183, upper bound: 0.7746994
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7747015, upper bound: 0.7785082
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3485284, 1.3529568
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6266239, 1.6255256
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6436327, 1.6331758
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600327, 2.1404798
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5339346, 1.5440493
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5610797, 1.5796766
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8532939, 1.8251181
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1061988, 2.1072364
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4586139, 1.4552751

Time for backsubstitution: 13.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1704

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7765147, upper bound: 0.7940208
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7962365, upper bound: 0.7743968
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3488889, 1.3513386
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6239364, 1.6261177
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6433198, 1.6332428
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600714, 2.1403065
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5330915, 1.5442338
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5592973, 1.5800710
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8508444, 1.8256557
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1047440, 2.1075501
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4552965, 1.4560025

Time for backsubstitution: 13.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1704

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1411

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7926450, upper bound: 0.7942900
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7965693, upper bound: 0.7932946
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3571444, 1.3501463
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6282132, 1.6259968
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6334889, 1.6478224
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403041, 2.1652527
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5501976, 1.5330858
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5875161, 1.5592976
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8275661, 1.8572617
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1119914, 2.1047442
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4639077, 1.4552965

Time for backsubstitution: 13.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2333

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7938362, upper bound: 0.7870316
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7848242, upper bound: 0.7958814
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3575048, 1.3485281
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6255257, 1.6265893
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6331761, 1.6478894
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403422, 2.1650791
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5493546, 1.5332704
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5857337, 1.5596919
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8251166, 1.8577995
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1105371, 2.1050580
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4605899, 1.4560242

Time for backsubstitution: 13.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7824012, upper bound: 0.7705149
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7687284, upper bound: 0.7842229
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3467226, 1.3547652
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6298630, 1.6222842
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6370513, 1.6397610
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443048, 2.1562064
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5406799, 1.5372975
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5766590, 1.5641072
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8326592, 1.8457513
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1060596, 2.1073766
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4543576, 1.4595308

Time for backsubstitution: 13.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1696

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2634

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7889877, upper bound: 0.7837654
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7873869, upper bound: 0.7854292
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3470831, 1.3531470
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6271751, 1.6228766
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6367385, 1.6398280
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443429, 2.1560330
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5398378, 1.5374825
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5748765, 1.5645015
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8302097, 1.8462889
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1046047, 2.1076903
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4510398, 1.4602582

Time for backsubstitution: 14.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7908227, upper bound: 0.7928994
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7965060, upper bound: 0.7871137
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3589473, 1.3483398
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6249678, 1.6292399
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6400721, 1.6412370
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1560359, 2.1495180
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5434389, 1.5398426
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5719464, 1.5748734
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8482056, 1.8366227
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1121287, 2.1046047
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4681649, 1.4510362

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7946507, upper bound: 0.7777895
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7666662, upper bound: 0.8055893
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3485289, 1.3529587
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6266177, 1.6255296
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6436355, 1.6331770
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600366, 2.1404743
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5339241, 1.5440540
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5610831, 1.5796831
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8532987, 1.8251121
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1061988, 2.1072371
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4586177, 1.4552705

Time for backsubstitution: 13.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2461

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7960965, upper bound: 0.8012395
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7919590, upper bound: 0.8007949
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3593082, 1.3467216
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6222804, 1.6298324
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6397603, 1.6413040
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1560740, 2.1493449
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5425959, 1.5400271
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5701640, 1.5752678
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8457561, 1.8371603
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1106744, 2.1049185
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4648471, 1.4517641

Time for backsubstitution: 13.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1704

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7737475, upper bound: 0.8045041
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7933694, upper bound: 0.7848041
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3488898, 1.3513405
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6239297, 1.6261221
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6433227, 1.6332440
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600747, 2.1403012
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5330811, 1.5442390
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5593007, 1.5800774
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8508492, 1.8256497
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1047444, 2.1075506
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4552999, 1.4559982

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2455

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7869047, upper bound: 0.7927178
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7875159, upper bound: 0.7927176
time: 4.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7914335, upper bound: 0.7982893
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.8038330, upper bound: 0.7858911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7817588, upper bound: 0.7712611
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7680794, upper bound: 0.7849572
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7964890, upper bound: 0.7834296
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7890189, upper bound: 0.7907789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7306726, upper bound: 0.7345219
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7306704, upper bound: 0.7345225
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7970597, upper bound: 0.7945283
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.8053996, upper bound: 0.7861005
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7896183, upper bound: 0.7746994
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7747015, upper bound: 0.7785082
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7765147, upper bound: 0.7940208
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7962365, upper bound: 0.7743968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7926450, upper bound: 0.7942900
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7965693, upper bound: 0.7932946
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7938362, upper bound: 0.7870316
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7848242, upper bound: 0.7958814
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7824012, upper bound: 0.7705149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7687284, upper bound: 0.7842229
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7889877, upper bound: 0.7837654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7873869, upper bound: 0.7854292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7908227, upper bound: 0.7928994
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7965060, upper bound: 0.7871137
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7946507, upper bound: 0.7777895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7666662, upper bound: 0.8055893
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7960965, upper bound: 0.8012395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7919590, upper bound: 0.8007949
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7737475, upper bound: 0.8045041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7933694, upper bound: 0.7848041
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7869047, upper bound: 0.7927178
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.11
Output dim: 6, lower bound: -0.7875159, upper bound: 0.7927176

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3285658, 1.3194742
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.5894709, 1.5949688
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6238387, 1.6473618
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1093037, 2.1265211
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5545082, 1.5393076
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5684118, 1.5428712
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.7604742, 1.7734759
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1090765, 2.1011014
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4559879, 1.4447491

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7776238, upper bound: 0.7704692
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7639432, upper bound: 0.7841665
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3264697, 1.3215704
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.5971994, 1.5872402
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6330316, 1.6381686
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1015623, 2.1342626
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5564356, 1.5373807
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5710802, 1.5402031
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.7437649, 1.7901847
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1083488, 2.1018291
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4533472, 1.4473898

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7899902, upper bound: 0.7580995
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7763127, upper bound: 0.7717979
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3585324, 1.3470905
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6220787, 1.6296492
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6394579, 1.6409549
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1518836, 2.1458404
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5372944, 1.5386944
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5718572, 1.5746791
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8415322, 1.8327289
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1121149, 2.1042480
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4664664, 1.4494956

Time for backsubstitution: 14.72 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.568244457244873
rel_dist={6: [-0.8170414609587353, 0.8170398270121066]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6042053, upper bound: 0.6006029
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6006064, upper bound: 0.6042042
time: 4.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.13
Output dim: 6, lower bound: -0.6042053, upper bound: 0.6006029
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.13
Output dim: 6, lower bound: -0.6006064, upper bound: 0.6042042

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2023723, 1.2023714
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4584968, 1.4584939
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5105715, 1.5105702
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395928, 1.9395957
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4179091, 1.4179034
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4536328, 1.4536359
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5256505, 1.5256486
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6721654, 1.6721683
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9464207, 1.9464207
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3582854, 1.3582876

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6041979, upper bound: 0.6003531
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6039549, upper bound: 0.6005955
time: 4.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2023728, 1.2023723
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4584939, 1.4584980
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5105705, 1.5105715
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395957, 1.9395931
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4179034, 1.4179082
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4536357, 1.4536328
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5256486, 1.5256536
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6721702, 1.6721654
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9464207, 1.9464211
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3582897, 1.3582854

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003560, upper bound: 0.6039544
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6003560, upper bound: 0.6042009
time: 4.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 6, lower bound: -0.6041979, upper bound: 0.6003531
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 6, lower bound: -0.6039549, upper bound: 0.6005955
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 6, lower bound: -0.6003560, upper bound: 0.6039544
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.16
Output dim: 6, lower bound: -0.6003560, upper bound: 0.6042009

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2020116, 1.2028196
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4592483, 1.4579012
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5106614, 1.5105032
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395547, 1.9396443
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4181457, 1.4177186
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4541302, 1.4532418
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5255039, 1.5258346
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6728525, 1.6716309
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9468341, 1.9461069
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3592165, 1.3575602

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5972089, upper bound: 0.6003338
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6041809, upper bound: 0.5972056
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2023723, 1.2020106
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4579046, 1.4584939
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5105050, 1.5105702
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395928, 1.9395576
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4177241, 1.4179034
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4532390, 1.4536359
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5256505, 1.5255020
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6716280, 1.6721683
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9461069, 1.9464207
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3575580, 1.3582876

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6038952, upper bound: 0.5959162
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5956848, upper bound: 0.5963338
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2020123, 1.2028205
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4592454, 1.4579055
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5106595, 1.5105044
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395580, 1.9396415
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4181399, 1.4177232
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4541330, 1.4532390
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5255020, 1.5258396
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6728573, 1.6716278
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9468341, 1.9461074
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3592212, 1.3575580

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5972089, upper bound: 0.6039366
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5972089, upper bound: 0.6008417
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2023728, 1.2020116
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4579017, 1.4584980
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5105031, 1.5105715
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395957, 1.9395549
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4177184, 1.4179082
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4532418, 1.4536328
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5256486, 1.5255067
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6716323, 1.6721654
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9461069, 1.9464211
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3575618, 1.3582854

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5960988, upper bound: 0.5959165
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5956848, upper bound: 0.6041384
time: 4.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.85 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.5972089, upper bound: 0.6003338
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.6041809, upper bound: 0.5972056
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.6038952, upper bound: 0.5959162
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.5956848, upper bound: 0.5963338
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.5972089, upper bound: 0.6039366
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.5972089, upper bound: 0.6008417
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.5960988, upper bound: 0.5959165
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.85
Output dim: 6, lower bound: -0.5956848, upper bound: 0.6041384

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2026036, 1.1982007
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4575994, 1.4581082
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5070984, 1.5109705
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9355545, 1.9401670
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4186914, 1.4135065
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4547527, 1.4484329
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5220311, 1.5262797
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6677604, 1.6722960
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9471684, 1.9434748
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3597579, 1.3533261

Time for backsubstitution: 12.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6010236, upper bound: 0.5956645
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928138, upper bound: 0.5960777
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1973927, 1.2028196
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4592483, 1.4562519
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5106614, 1.5069405
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395547, 1.9356439
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4139335, 1.4177186
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4493210, 1.4532418
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5255039, 1.5223618
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6728525, 1.6665387
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9442019, 1.9461069
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3549824, 1.3575602

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6041249, upper bound: 0.5925780
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928138, upper bound: 0.5929050
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1997619, 1.2003052
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4548461, 1.4538103
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5010102, 1.5043678
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9169235, 1.9247549
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4113636, 1.4081645
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4385819, 1.4311914
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5117993, 1.5042880
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6418810, 1.6527433
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9459038, 1.9462872
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3514106, 1.3542728

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6007887, upper bound: 0.5958993
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6038819, upper bound: 0.5928139
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2006650, 1.1993999
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4532216, 1.4554331
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5042999, 1.5010759
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247894, 1.9168878
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4079857, 1.4115429
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4307938, 1.4389729
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5044365, 1.5116470
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6522007, 1.6424217
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9459734, 1.9462175
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3535411, 1.3521402

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5963104
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5931344
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2026043, 1.1982017
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4575956, 1.4581122
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5070965, 1.5109717
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9355578, 1.9401631
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4186857, 1.4135113
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4547555, 1.4484296
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5220292, 1.5262847
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6677651, 1.6722910
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9471684, 1.9434750
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3597622, 1.3533237

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5931349, upper bound: 0.5956645
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6038819
time: 7.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1973934, 1.2028205
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4592454, 1.4562559
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5106595, 1.5069417
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395580, 1.9356413
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4139278, 1.4177232
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4493239, 1.4532390
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5255020, 1.5223668
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6728573, 1.6665356
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9442019, 1.9461074
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3549867, 1.3575580

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5963106, upper bound: 0.5925778
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5959000, upper bound: 0.6007885
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1997626, 1.2003043
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4548409, 1.4538147
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5010087, 1.5043691
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9169269, 1.9247510
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4113579, 1.4081697
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4385848, 1.4311883
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5117936, 1.5042922
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6418858, 1.6527383
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9459038, 1.9462876
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3514140, 1.3542681

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5929024, upper bound: 0.5959014
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5960751, upper bound: 0.5928106
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2006657, 1.1994009
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4532182, 1.4554374
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5043008, 1.5010772
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247928, 1.9168851
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4079800, 1.4115481
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4307971, 1.4389763
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5044346, 1.5116513
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6522055, 1.6424186
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9459734, 1.9462180
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3535440, 1.3521380

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6041270
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6010236
time: 4.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.6010236, upper bound: 0.5956645
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5928138, upper bound: 0.5960777
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.6041249, upper bound: 0.5925780
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5928138, upper bound: 0.5929050
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.6007887, upper bound: 0.5958993
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.6038819, upper bound: 0.5928139
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5963104
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5931344
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5931349, upper bound: 0.5956645
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6038819
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5963106, upper bound: 0.5925778
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5959000, upper bound: 0.6007885
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5929024, upper bound: 0.5959014
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5960751, upper bound: 0.5928106
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6041270
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.52
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6010236

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1999929, 1.1964953
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4545405, 1.4534249
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4976037, 1.5047688
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9128852, 1.9253645
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4123325, 1.4037683
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4400918, 1.4259875
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5081799, 1.5050657
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6380129, 1.6528680
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9469657, 1.9433415
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3536105, 1.3493111

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6001961, upper bound: 0.5955231
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6008718, upper bound: 0.5948498
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2008946, 1.1955903
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4529159, 1.4550465
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5008929, 1.5014762
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9207511, 1.9174974
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4089527, 1.4071465
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4323070, 1.4337690
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5008171, 1.5124240
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6483326, 1.6425486
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9470339, 1.9432716
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3557386, 1.3471787

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2495

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5864681, upper bound: 0.5864556
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5830053, upper bound: 0.5896969
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1947820, 1.2011142
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4561899, 1.4515686
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5011666, 1.5007381
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9168849, 1.9208415
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4075737, 1.4079800
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4346635, 1.4307971
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5116527, 1.5011477
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6431060, 1.6471128
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9439993, 1.9459734
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3488350, 1.3535452

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1769

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1411

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6029195, upper bound: 0.5920207
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6035913, upper bound: 0.5913804
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1956854, 1.2002089
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4545653, 1.4531914
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5044558, 1.4974462
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247508, 1.9129744
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4041948, 1.4113584
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4268754, 1.4385786
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5042899, 1.5085068
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6534257, 1.6367910
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9440689, 1.9459038
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3509650, 1.3514128

Time for backsubstitution: 12.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5855562, upper bound: 0.5864213
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5890741, upper bound: 0.5858828
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2003539, 1.1956861
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4531963, 1.4540172
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4974473, 1.5048358
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9129233, 1.9252777
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4119110, 1.4039528
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4392006, 1.4263818
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5083268, 1.5047331
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6367879, 1.6534057
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9462380, 1.9436560
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3519516, 1.3500388

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1837

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5916518, upper bound: 0.5909081
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5958028, upper bound: 0.5949925
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1951430, 1.2003052
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4548461, 1.4521608
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5010102, 1.5008051
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9169235, 1.9207549
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4071522, 1.4081645
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4337723, 1.4311914
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5117993, 1.5008149
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6418810, 1.6476502
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9432721, 1.9462872
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3471766, 1.3542728

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1383

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5906000, upper bound: 0.5912202
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6022936, upper bound: 0.5795307
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2012553, 1.1947811
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4515717, 1.4556386
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5007365, 1.5015432
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9207892, 1.9174106
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4085312, 1.4073310
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4314158, 1.4341636
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5009639, 1.5120914
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6471076, 1.6430860
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9463067, 1.9435863
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3540802, 1.3479064

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2342

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5827613, upper bound: 0.5950191
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5912904, upper bound: 0.5865027
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1960461, 1.1993999
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4532216, 1.4537835
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5042999, 1.4975132
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247894, 1.9128876
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4037733, 1.4115429
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4259841, 1.4389729
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5044365, 1.5081742
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6522007, 1.6373286
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9433417, 1.9462175
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3493066, 1.3521402

Time for backsubstitution: 12.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1411

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5913844, upper bound: 0.5925928
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5920237, upper bound: 0.5919485
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1999938, 1.1964943
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4545352, 1.4534290
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4976027, 1.5047700
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9128881, 1.9253607
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4123268, 1.4037733
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4400952, 1.4259844
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5081742, 1.5050697
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6380172, 1.6528628
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9469652, 1.9433417
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3536134, 1.3493063

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2634

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5873711, upper bound: 0.5881876
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5855369, upper bound: 0.5899875
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2008953, 1.1955912
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4529126, 1.4550506
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5008948, 1.5014774
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9207540, 1.9174936
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4089470, 1.4071515
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4323103, 1.4337723
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5008152, 1.5124283
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6483369, 1.6425433
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9470339, 1.9432721
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3557420, 1.3471763

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1769

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928106, upper bound: 0.5970481
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5864536, upper bound: 0.6038801
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1947830, 1.2011132
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4561851, 1.4515727
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5011652, 1.5007393
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9168887, 1.9208376
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4075680, 1.4079847
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4346664, 1.4307940
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5116470, 1.5011518
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6431103, 1.6471076
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9439993, 1.9459741
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3488383, 1.3535407

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5832286, upper bound: 0.5887115
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5924078, upper bound: 0.5794069
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1956861, 1.2002099
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4545619, 1.4531955
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5044572, 1.4974474
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247546, 1.9129717
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4041901, 1.4113631
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4268787, 1.4385819
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5042880, 1.5085111
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6534300, 1.6367879
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9440689, 1.9459043
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3509688, 1.3514104

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1837

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1389

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928681, upper bound: 0.5976578
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5894218, upper bound: 0.5978017
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2003546, 1.1956851
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4531915, 1.4540215
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4974463, 1.5048370
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9129262, 1.9252741
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4119053, 1.4039578
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4392040, 1.4263787
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5083210, 1.5047371
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6367927, 1.6534004
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9462380, 1.9436555
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3519549, 1.3500342

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2495

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5910896, upper bound: 0.5926562
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5897152, upper bound: 0.5940194
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1951437, 1.2003043
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4548409, 1.4521651
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5010087, 1.5008063
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9169269, 1.9207511
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4071465, 1.4081697
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4337752, 1.4311883
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5117936, 1.5008192
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6418858, 1.6476452
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9432721, 1.9462876
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3471799, 1.3542681

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2455

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5896840, upper bound: 0.5909529
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5917171, upper bound: 0.5892785
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2012560, 1.1947820
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4515688, 1.4556431
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5007384, 1.5015444
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9207921, 1.9174070
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4085255, 1.4073360
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4314191, 1.4341667
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5009620, 1.5120955
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6471124, 1.6430809
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9463067, 1.9435859
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3540835, 1.3479042

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5874929, upper bound: 0.6033386
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5917924, upper bound: 0.5990605
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1960468, 1.1994009
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4532182, 1.4537879
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5043008, 1.4975144
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247928, 1.9128852
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4037685, 1.4115481
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4259875, 1.4389763
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5044346, 1.5081782
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6522055, 1.6373255
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9433417, 1.9462180
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3493104, 1.3521380

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1704

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5800879, upper bound: 0.5994587
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5940745, upper bound: 0.5885381
time: 4.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.6001961, upper bound: 0.5955231
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.6008718, upper bound: 0.5948498
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5864681, upper bound: 0.5864556
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5830053, upper bound: 0.5896969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.6029195, upper bound: 0.5920207
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.6035913, upper bound: 0.5913804
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5855562, upper bound: 0.5864213
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5890741, upper bound: 0.5858828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5916518, upper bound: 0.5909081
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5958028, upper bound: 0.5949925
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5906000, upper bound: 0.5912202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.6022936, upper bound: 0.5795307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5827613, upper bound: 0.5950191
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5912904, upper bound: 0.5865027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5913844, upper bound: 0.5925928
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5920237, upper bound: 0.5919485
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5873711, upper bound: 0.5881876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5855369, upper bound: 0.5899875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5928106, upper bound: 0.5970481
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5864536, upper bound: 0.6038801
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5832286, upper bound: 0.5887115
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5924078, upper bound: 0.5794069
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5928681, upper bound: 0.5976578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5894218, upper bound: 0.5978017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5910896, upper bound: 0.5926562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5897152, upper bound: 0.5940194
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5896840, upper bound: 0.5909529
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5917171, upper bound: 0.5892785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5874929, upper bound: 0.6033386
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5917924, upper bound: 0.5990605
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5800879, upper bound: 0.5994587
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.33
Output dim: 6, lower bound: -0.5940745, upper bound: 0.5885381

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2017713, 1.1979833
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4463768, 1.4467022
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4553986, 1.4671035
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9228241, 1.9390068
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4114821, 1.4021282
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3865387, 1.3660681
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5362282, 1.5374637
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6611471, 1.6731603
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8945642, 1.8997681
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3513417, 1.3467677

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2495

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5984319, upper bound: 0.5923001
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5970644, upper bound: 0.5936631
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2014811, 1.1982734
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4478173, 1.4452616
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4599385, 1.4625638
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9265273, 1.9353037
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4106925, 1.4029183
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3801725, 1.3724344
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5405779, 1.5331142
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6583047, 1.6760023
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9033918, 1.8909402
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3510671, 1.3470423

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 422

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5782567, upper bound: 0.5721410
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5782567, upper bound: 0.5721410
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1628683, 1.1697538
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3885720, 1.3979222
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4762430, 1.4732553
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8528895, 1.8420732
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3661711, 1.3712842
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3575346, 1.3552752
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4799094, 1.4840293
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6370239, 1.6297939
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9163651, 1.9103482
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3237915, 1.3197036

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1704

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2342

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5765613, upper bound: 0.5851646
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5851770, upper bound: 0.5767676
time: 5.03 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.43
Output dim: 6, lower bound: -0.5984319, upper bound: 0.5923001
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.43
Output dim: 6, lower bound: -0.5970644, upper bound: 0.5936631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.43
Output dim: 6, lower bound: -0.5782567, upper bound: 0.5721410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.43
Output dim: 6, lower bound: -0.5782567, upper bound: 0.5721410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.43
Output dim: 6, lower bound: -0.5765613, upper bound: 0.5851646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.43
Output dim: 6, lower bound: -0.5851770, upper bound: 0.5767676
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5830053, upper bound: 0.5896969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.6029195, upper bound: 0.5920207
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.6035913, upper bound: 0.5913804
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5855562, upper bound: 0.5864213
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5890741, upper bound: 0.5858828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5916518, upper bound: 0.5909081
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5958028, upper bound: 0.5949925
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5906000, upper bound: 0.5912202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.6022936, upper bound: 0.5795307
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5827613, upper bound: 0.5950191
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5912904, upper bound: 0.5865027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5913844, upper bound: 0.5925928
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5920237, upper bound: 0.5919485
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5873711, upper bound: 0.5881876
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5855369, upper bound: 0.5899875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5928106, upper bound: 0.5970481
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5864536, upper bound: 0.6038801
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5832286, upper bound: 0.5887115
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5924078, upper bound: 0.5794069
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5928681, upper bound: 0.5976578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5894218, upper bound: 0.5978017
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5910896, upper bound: 0.5926562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5897152, upper bound: 0.5940194
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5896840, upper bound: 0.5909529
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5917171, upper bound: 0.5892785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5874929, upper bound: 0.6033386
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5917924, upper bound: 0.5990605
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5800879, upper bound: 0.5994587
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.43
Output dim: 6, lower bound: -0.5940745, upper bound: 0.5885381
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.525653600692749
rel_dist={6: [-0.6052081522674859, 0.6052060697814263]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4477134
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4479308
time: 4.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.10
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4477134
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.10
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4479308

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1007192, 1.1009889
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3445458, 1.3440979
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4200699, 1.4200180
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7879472, 1.7879763
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3316553, 1.3315146
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3646636, 1.3643665
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4516771, 1.4517879
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5464911, 1.5460830
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8387947, 1.8385522
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2888753, 1.2883222

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466573, upper bound: 0.4477110
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466572, upper bound: 0.4466554
time: 4.41 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1009889, 1.1007192
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3440981, 1.3445458
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4200180, 1.4200701
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7879763, 1.7879472
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3315146, 1.3316550
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3643665, 1.3646636
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4517879, 1.4516771
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5460830, 1.5464911
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8385520, 1.8387947
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2883222, 1.2888751

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445123, upper bound: 0.4447265
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445124, upper bound: 0.4479287
time: 4.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 6, lower bound: -0.4466573, upper bound: 0.4477110
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 6, lower bound: -0.4466572, upper bound: 0.4466554
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 6, lower bound: -0.4445123, upper bound: 0.4447265
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.31
Output dim: 6, lower bound: -0.4445124, upper bound: 0.4479287

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0978372, 1.0963700
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3428965, 1.3430672
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4165075, 1.4177985
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7839475, 1.7854836
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3290291, 1.3273027
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3616650, 1.3595574
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4482043, 1.4496210
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5413990, 1.5429091
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8371511, 1.8359199
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2862329, 1.2840881

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4445080
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4477111
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0961003, 1.0981069
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3435154, 1.3424485
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4178507, 1.4164553
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7854543, 1.7839763
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3274431, 1.3288887
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3598545, 1.3613679
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4495101, 1.4483151
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5433173, 1.5409908
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8361626, 1.8369086
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2846413, 1.2856798

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4450990, upper bound: 0.4450990
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4450990, upper bound: 0.4466546
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0983784, 1.0984097
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3399553, 1.3398626
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4105237, 1.4116731
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7653069, 1.7678998
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3229024, 1.3219166
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3445172, 1.3422186
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4330263, 1.4304624
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5163364, 1.5201845
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8383493, 1.8386149
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2821739, 1.2834370

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4445119, upper bound: 0.4447257
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4446545, upper bound: 0.4447254
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0986795, 1.0981085
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3394146, 1.3404036
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4116213, 1.4105757
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7679291, 1.7652779
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3217762, 1.3230429
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3419213, 1.3448145
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4305732, 1.4329154
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5197763, 1.5167446
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8383722, 1.8385916
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2828844, 1.2827270

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4479258
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4468682
time: 4.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4445080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4477111
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4450990, upper bound: 0.4450990
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4450990, upper bound: 0.4466546
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4445119, upper bound: 0.4447257
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4446545, upper bound: 0.4447254
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4479258
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4468682

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0952268, 1.0940604
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3387537, 1.3383839
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4070132, 1.4094018
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7612777, 1.7654362
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3204174, 1.3175647
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3418143, 1.3371119
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4294424, 1.4284062
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5116515, 1.5166016
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369479, 1.8357401
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2800841, 1.2786498

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4445071
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4438018, upper bound: 0.4445066
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0955272, 1.0937593
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3382130, 1.3389244
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4081104, 1.4083042
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7638993, 1.7628136
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3192911, 1.3186908
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3392198, 1.3397079
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4269893, 1.4308591
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5150914, 1.5131617
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369708, 1.8357167
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2807937, 1.2779398

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4446534
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4477079
time: 4.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0960996, 1.0981059
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3435118, 1.3424442
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4178498, 1.4164541
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7854509, 1.7839737
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3274400, 1.3288839
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3598511, 1.3613653
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4495056, 1.4483101
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5433130, 1.5409873
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8361621, 1.8369083
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2846370, 1.2856755

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4434765
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434764, upper bound: 0.4435855
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0960994, 1.0981061
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3435113, 1.3424454
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4178498, 1.4164544
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7854524, 1.7839727
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3274391, 1.3288858
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3598521, 1.3613648
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4495053, 1.4483109
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5433145, 1.5409863
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8361621, 1.8369083
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2846375, 1.2856753

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4448696, upper bound: 0.4434739
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4466499
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0983775, 1.0984092
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3399532, 1.3398585
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4105232, 1.4116719
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7653031, 1.7678976
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3228998, 1.3219118
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3445144, 1.3422165
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4330239, 1.4304581
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5163317, 1.5201817
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8383484, 1.8386142
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2821710, 1.2834349

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434764, upper bound: 0.4447245
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4436911
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0983772, 1.0984089
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3399513, 1.3398595
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4105227, 1.4116714
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7653041, 1.7678964
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3228979, 1.3219137
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3445134, 1.3422155
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4330220, 1.4304588
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5163326, 1.5201797
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8383484, 1.8386140
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2821710, 1.2834334

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4435857, upper bound: 0.4447228
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4446536, upper bound: 0.4436900
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0957971, 1.0934896
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3377652, 1.3393724
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4080584, 1.4083564
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7639284, 1.7627847
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3191504, 1.3188312
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3389223, 1.3400049
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4271002, 1.4307482
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5146832, 1.5135698
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8367281, 1.8359594
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2802415, 1.2784929

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4448724
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4479254
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0940604, 1.0952268
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3383837, 1.3387539
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4094021, 1.4070131
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7654362, 1.7612774
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3175645, 1.3204174
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3371122, 1.3418145
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4284062, 1.4294424
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5166016, 1.5116515
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8357401, 1.8369479
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2786498, 1.2800846

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4438026
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4468670
time: 4.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4445071
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4438018, upper bound: 0.4445066
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4446534
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4477079
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4434765
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434764, upper bound: 0.4435855
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4448696, upper bound: 0.4434739
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4466499
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434764, upper bound: 0.4447245
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4436911
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4435857, upper bound: 0.4447228
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4446536, upper bound: 0.4436900
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4448724
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4479254
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4438026
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.99
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4468670

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0952260, 1.0940599
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3387516, 1.3383799
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4070122, 1.4094006
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7612743, 1.7654343
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3204145, 1.3175597
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3418114, 1.3371098
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4294403, 1.4284022
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5116467, 1.5165985
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369479, 1.8357399
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2800813, 1.2786479

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4353902, upper bound: 0.4363426
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4353903, upper bound: 0.4380900
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0952256, 1.0940597
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3387496, 1.3383806
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4070122, 1.4094001
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7612753, 1.7654328
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3204126, 1.3175616
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3418100, 1.3371089
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4294384, 1.4284022
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5116477, 1.5165968
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369479, 1.8357399
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2800813, 1.2786462

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2804

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2333

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4431466, upper bound: 0.4425314
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4418224, upper bound: 0.4438509
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0955265, 1.0937583
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3382099, 1.3389204
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4081089, 1.4083030
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7638965, 1.7628117
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3192883, 1.3186860
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3392165, 1.3397038
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4269860, 1.4308548
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5150867, 1.5131588
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369708, 1.8357167
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2807903, 1.2779372

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 578

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4402020, upper bound: 0.4413301
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4402020, upper bound: 0.4413301
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0955267, 1.0937586
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3382089, 1.3389218
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4081089, 1.4083031
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7638974, 1.7628107
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3192863, 1.3186877
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3392169, 1.3397048
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4269853, 1.4308567
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5150886, 1.5131569
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369708, 1.8357167
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2807922, 1.2779362

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1116

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1837

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4374747, upper bound: 0.4421109
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4381214, upper bound: 0.4417480
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0934887, 1.0957961
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3393686, 1.3377621
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4083550, 1.4080566
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7627831, 1.7639251
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3188267, 1.3191476
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3400009, 1.3389194
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4307439, 1.4270968
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5135670, 1.5146785
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8359590, 1.8367281
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2784901, 1.2802374

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4406697, upper bound: 0.4408896
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4422725, upper bound: 0.4406660
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0937903, 1.0954955
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3388278, 1.3383036
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4094527, 1.4069601
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7654052, 1.7613034
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3177004, 1.3202739
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3374069, 1.3415143
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4282911, 1.4295511
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5170069, 1.5112386
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8359823, 1.8367052
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2792010, 1.2795279

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4296326, upper bound: 0.4328532
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4296326, upper bound: 0.4329125
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0954957, 1.0937903
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3383033, 1.3388278
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4069598, 1.4094527
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7613034, 1.7654052
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3202744, 1.3177004
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3415143, 1.3374069
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4295511, 1.4282913
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5112386, 1.5170069
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8367057, 1.8359823
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2795281, 1.2792008

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2804

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1389

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4416059, upper bound: 0.4428444
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4416059, upper bound: 0.4428582
time: 7.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0954952, 1.0937901
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3383019, 1.3388286
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4069598, 1.4094523
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7613044, 1.7654040
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3202724, 1.3177021
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3415129, 1.3374059
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4295492, 1.4282913
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5112395, 1.5170050
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8367057, 1.8359823
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2795281, 1.2791994

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1248

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1978

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4429061, upper bound: 0.4444290
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4432904, upper bound: 0.4441524
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0937583, 1.0955265
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3389204, 1.3382101
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4083035, 1.4081087
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7628117, 1.7638965
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3186855, 1.3192880
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3397038, 1.3392165
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4308550, 1.4269860
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5131588, 1.5150867
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8357167, 1.8369706
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2779374, 1.2807903

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2326

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1837

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4386827, upper bound: 0.4381179
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390454, upper bound: 0.4374714
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0957961, 1.0934887
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3377621, 1.3393683
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4080565, 1.4083552
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7639251, 1.7627831
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3191471, 1.3188262
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3389194, 1.3400009
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4270968, 1.4307439
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5146785, 1.5135670
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8367281, 1.8359590
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2802377, 1.2784898

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2495

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 626

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4419124, upper bound: 0.4445761
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4431800, upper bound: 0.4433051
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0957966, 1.0934889
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3377612, 1.3393700
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4080575, 1.4083552
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7639265, 1.7627816
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3191452, 1.3188281
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3389199, 1.3400018
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4270961, 1.4307458
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5146804, 1.5135653
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8367286, 1.8359590
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2802391, 1.2784894

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4395146, upper bound: 0.4439647
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4395147, upper bound: 0.4457171
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0940599, 1.0952258
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3383796, 1.3387516
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4094002, 1.4070122
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7654343, 1.7612743
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3175602, 1.3204145
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3371098, 1.3418114
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4284022, 1.4294403
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5165987, 1.5116467
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8357401, 1.8369479
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2786479, 1.2800810

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4417712, upper bound: 0.4449439
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4417713, upper bound: 0.4461900
time: 6.84 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4353902, upper bound: 0.4363426
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4353903, upper bound: 0.4380900
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4431466, upper bound: 0.4425314
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4418224, upper bound: 0.4438509
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4402020, upper bound: 0.4413301
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4402020, upper bound: 0.4413301
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4374747, upper bound: 0.4421109
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4381214, upper bound: 0.4417480
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4406697, upper bound: 0.4408896
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4422725, upper bound: 0.4406660
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4296326, upper bound: 0.4328532
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4296326, upper bound: 0.4329125
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4416059, upper bound: 0.4428444
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4416059, upper bound: 0.4428582
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4429061, upper bound: 0.4444290
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4432904, upper bound: 0.4441524
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4386827, upper bound: 0.4381179
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4390454, upper bound: 0.4374714
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4419124, upper bound: 0.4445761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4431800, upper bound: 0.4433051
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4395146, upper bound: 0.4439647
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4395147, upper bound: 0.4457171
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4417712, upper bound: 0.4449439
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.54
Output dim: 6, lower bound: -0.4417713, upper bound: 0.4461900

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0305495, 1.0294476
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.2493272, 1.2506936
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3392012, 1.3442557
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7197080, 1.7238672
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3232474, 1.3203969
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.2881200, 1.2813890
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4146671, 1.4133539
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.4516280, 1.4570043
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8351288, 1.8327959
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2678165, 1.2665632

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1837

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4354771, upper bound: 0.4382035
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4361440, upper bound: 0.4377899
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0961301, 1.0943706
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3202314, 1.3203464
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3894260, 1.3930308
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7451267, 1.7510717
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3207078, 1.3178966
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3396811, 1.3355215
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4222794, 1.4210310
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5117884, 1.5175598
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8171043, 1.8173280
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2683005, 1.2678361

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4414291, upper bound: 0.4427736
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4412516, upper bound: 0.4428404
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0960758, 1.0944250
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3198195, 1.3207581
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3905382, 1.3919184
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7469716, 1.7492266
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3204665, 1.3181381
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3396287, 1.3355737
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4222889, 1.4210215
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5117941, 1.5175536
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8180513, 1.8163810
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2681651, 1.2679718

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4418298, upper bound: 0.4430734
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4418298, upper bound: 0.4432547
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0974050, 1.0953741
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3335238, 1.3352070
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4106126, 1.4111214
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7580199, 1.7556405
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3193173, 1.3189890
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3329942, 1.3341427
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4186513, 1.4227471
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5168657, 1.5156898
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8336139, 1.8327694
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2800934, 1.2782977

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1116

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4407165, upper bound: 0.4400125
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4382776, upper bound: 0.4436455
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0577700, 1.0595398
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.2734172, 1.2774515
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3810406, 1.3801343
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.6960649, 1.6924183
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2809842, 1.2829659
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.2616818, 1.2615080
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4012084, 1.4023511
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5033717, 1.5017805
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8060598, 1.8045473
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2482920, 1.2480423

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 1789

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1389

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4391059, upper bound: 0.4415911
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4391355, upper bound: 0.4421868
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0618334, 1.0554626
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.2758238, 1.2750263
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3798361, 1.3813245
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.6935444, 1.6949198
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2832835, 1.2806594
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.2604260, 1.2627485
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.3987012, 1.4048469
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5028896, 1.5022569
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8053079, 1.8052907
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2497826, 1.2465420

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4260433, upper bound: 0.4319536
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4260432, upper bound: 0.4319858
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0948918, 1.0958049
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3345931, 1.3351126
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4085081, 1.4060380
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7655661, 1.7618806
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3179336, 1.3208880
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3362627, 1.3409536
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4262133, 1.4270086
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5136557, 1.5086899
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8440323, 1.8446360
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2764552, 1.2779241

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2307
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1837

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4380871, upper bound: 0.4392495
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4384498, upper bound: 0.4391619
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0946624, 1.0960579
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3347657, 1.3349648
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4084265, 1.4061316
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7660401, 1.7614419
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3180337, 1.3208218
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3362598, 1.3409641
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4259706, 1.4272773
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5136414, 1.5087619
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8434820, 1.8452401
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2765105, 1.2778883

Time for backsubstitution: 12.52 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.4518237113952637
rel_dist={6: [-0.44793373691976335, 0.4479324567017895]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 2411.05 seconds
