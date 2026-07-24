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
execution time: IAR + LP analysis = 13.11 + 32.11 = 45.21 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.79 seconds, max iter: 100)

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
Binary search time: 144.19 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 3410.60 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8170355, upper bound: 0.8170341
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8170355, upper bound: 0.8170341
time: 4.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.99
Output dim: 6, lower bound: -0.8170355, upper bound: 0.8170341
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.99
Output dim: 6, lower bound: -0.8170355, upper bound: 0.8170341

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3539519, 1.3555698
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6313047, 1.6286170
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6465466, 1.6462340
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1669741, 2.1671474
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5478785, 1.5470359
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5883374, 1.5865552
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8624063, 1.8599565
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1088943, 2.1074400
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4647396, 1.4614222

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8142981, upper bound: 0.8170156
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8142981, upper bound: 0.8142950
time: 4.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3543127, 1.3539519
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6286173, 1.6292095
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6462343, 1.6463012
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1670117, 2.1669741
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5470359, 1.5472209
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5865550, 1.5869491
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8599563, 1.8604941
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1074400, 2.1077538
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4614227, 1.4621496

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8142981, upper bound: 0.8170156
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8170147, upper bound: 0.8142953
time: 4.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.55
Output dim: 6, lower bound: -0.8142981, upper bound: 0.8170156
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.55
Output dim: 6, lower bound: -0.8142981, upper bound: 0.8142950
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.55
Output dim: 6, lower bound: -0.8142981, upper bound: 0.8170156
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.55
Output dim: 6, lower bound: -0.8170147, upper bound: 0.8142953

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3597550, 1.3509512
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6296554, 1.6306801
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6429842, 1.6507313
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1629739, 2.1721909
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5531821, 1.5428240
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5943921, 1.5817461
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8573136, 1.8663750
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1121945, 2.1048079
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4700561, 1.4571881

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956287, upper bound: 0.7983432
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956287, upper bound: 0.8169949
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3493333, 1.3555698
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6313047, 1.6269674
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6465466, 1.6426713
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1669741, 2.1631472
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5436668, 1.5470359
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5835283, 1.5865552
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8624063, 1.8548644
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1062622, 2.1074400
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4605055, 1.4614222

Time for backsubstitution: 12.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8169939, upper bound: 0.7956267
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983431, upper bound: 0.8142749
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3601153, 1.3493330
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6269674, 1.6312726
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6426713, 1.6507984
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1630120, 2.1720178
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5523391, 1.5430083
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5926096, 1.5821404
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8548646, 1.8669126
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1107397, 2.1051216
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4667387, 1.4579158

Time for backsubstitution: 12.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956287, upper bound: 0.7983432
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956287, upper bound: 0.8169949
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3496935, 1.3539519
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6286173, 1.6275599
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6462343, 1.6427385
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1670117, 2.1629741
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5428243, 1.5472209
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5817459, 1.5869491
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8599563, 1.8554020
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1048079, 2.1077538
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4571881, 1.4621496

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8169939, upper bound: 0.7956267
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983431, upper bound: 0.8142749
time: 4.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.7956287, upper bound: 0.7983432
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.7956287, upper bound: 0.8169949
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.8169939, upper bound: 0.7956267
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.7983431, upper bound: 0.8142749
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.7956287, upper bound: 0.7983432
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.7956287, upper bound: 0.8169949
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.8169939, upper bound: 0.7956267
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -0.7983431, upper bound: 0.8142749

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3571444, 1.3501470
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6282177, 1.6259968
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6334898, 1.6478224
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403041, 2.1652555
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5502024, 1.5330858
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5875161, 1.5593007
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8275661, 1.8572664
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1119914, 2.1047442
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4639077, 1.4553001

Time for backsubstitution: 13.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3589473, 1.3483405
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6249723, 1.6292399
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6400740, 1.6412370
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1560359, 2.1495214
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5434437, 1.5398426
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5719464, 1.5748765
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8482056, 1.8366275
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1121287, 2.1046047
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4681649, 1.4510398

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066097
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3467226, 1.3547659
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6298666, 1.6222842
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6370528, 1.6397610
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443048, 2.1562095
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5406852, 1.5372975
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5766590, 1.5641103
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8326592, 1.8457561
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1060591, 2.1073766
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4543576, 1.4595344

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3485289, 1.3529594
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6266212, 1.6255296
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6436369, 1.6331770
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600366, 2.1404777
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5339289, 1.5440540
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5610831, 1.5796862
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8532987, 1.8251166
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1061988, 2.1072371
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4586177, 1.4552741

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983427, upper bound: 0.7962784
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7983427, upper bound: 0.8038854
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3575048, 1.3485291
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6255302, 1.6265893
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6331770, 1.6478894
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1403422, 2.1650822
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5493593, 1.5332704
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5857337, 1.5596952
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8251166, 1.8578038
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1105366, 2.1050580
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4605899, 1.4560277

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3593082, 1.3467224
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6222839, 1.6298324
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6397612, 1.6413040
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1560740, 2.1493478
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5426011, 1.5400271
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5701640, 1.5752711
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8457561, 1.8371649
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1106739, 2.1049185
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4648471, 1.4517677

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066104
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3470831, 1.3531477
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6271791, 1.6228766
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6367400, 1.6398280
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1443429, 2.1560359
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5398426, 1.5374825
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5748765, 1.5645046
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8302097, 1.8462934
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1046047, 2.1076903
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4510398, 1.4602618

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3488898, 1.3513415
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6239338, 1.6261221
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6433241, 1.6332440
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1600747, 2.1403041
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5330858, 1.5442390
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5593007, 1.5800805
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8508492, 1.8256543
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.1047440, 2.1075506
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4552999, 1.4560015

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7962775
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8038851
time: 4.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066097
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7983427, upper bound: 0.7962784
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7983427, upper bound: 0.8038854
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.8038854, upper bound: 0.7983430
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7962783, upper bound: 0.7983430
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7990869
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8066104
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.8066092, upper bound: 0.7956263
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7990860, upper bound: 0.7956264
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7956283, upper bound: 0.7962775
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 6, lower bound: -0.7956282, upper bound: 0.8038851

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

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7793057, upper bound: 0.7909248
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7965727, upper bound: 0.7740640
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7717091, upper bound: 0.7909248
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7889549, upper bound: 0.7740639
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7916702
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7748070
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7992129
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7823289
time: 4.74 seconds

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

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7823284, upper bound: 0.7883014
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7992124, upper bound: 0.7710582
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7748066, upper bound: 0.7883017
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7916701, upper bound: 0.7710580
time: 4.46 seconds

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

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7889545
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7717088
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7965727
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7793063
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7793057, upper bound: 0.7909248
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7965727, upper bound: 0.7740640
time: 4.89 seconds

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

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7717091, upper bound: 0.7909248
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7889549, upper bound: 0.7740639
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7916703
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7748070
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7992130
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7823289
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7823284, upper bound: 0.7883014
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7992124, upper bound: 0.7710582
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7748066, upper bound: 0.7883017
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7916701, upper bound: 0.7710580
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7889545
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7717088
time: 4.48 seconds

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

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7965727
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7793063
time: 4.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7793057, upper bound: 0.7909248
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7965727, upper bound: 0.7740640
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7717091, upper bound: 0.7909248
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7889549, upper bound: 0.7740639
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7916702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7748070
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7992129
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7823289
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7823284, upper bound: 0.7883014
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7992124, upper bound: 0.7710582
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7748066, upper bound: 0.7883017
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7916701, upper bound: 0.7710580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7889545
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7717088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7965727
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7793063
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7793057, upper bound: 0.7909248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7965727, upper bound: 0.7740640
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7717091, upper bound: 0.7909248
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7889549, upper bound: 0.7740639
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7916703
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7748070
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7992130
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7883016, upper bound: 0.7823289
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7823284, upper bound: 0.7883014
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7992124, upper bound: 0.7710582
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7748066, upper bound: 0.7883017
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7916701, upper bound: 0.7710580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7889545
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7717088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7710602, upper bound: 0.7965727
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.57
Output dim: 6, lower bound: -0.7909248, upper bound: 0.7793063

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3569977, 1.3499005
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6277821, 1.6256849
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6203430, 1.6339529
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1379499, 2.1659827
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5215530, 1.4961249
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5696986, 1.5426271
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8139780, 1.8475659
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.0900140, 2.0911133
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4639249, 1.4542797

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7564543, upper bound: 0.7770731
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7648954, upper bound: 0.7769549
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.3568962, 1.3500021
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.6279156, 1.6255515
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.6196229, 1.6346729
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -2.1410241, 2.1629088
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.5132523, 1.5044253
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.5708358, 1.5414896
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5682445, 1.5682445
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.8178556, 1.8436882
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -2.0983610, 2.0827661
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.4628782, 1.4553263

Time for backsubstitution: 14.35 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.568244457244873
rel_dist={6: [-0.8170414609587353, 0.8170398270121066]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6049578, upper bound: 0.6049558
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6049578, upper bound: 0.6051971
time: 4.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.81
Output dim: 6, lower bound: -0.6049578, upper bound: 0.6049558
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.81
Output dim: 6, lower bound: -0.6049578, upper bound: 0.6051971

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2020123, 1.2028213
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4592495, 1.4579055
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5106609, 1.5105044
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395580, 1.9396448
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4181445, 1.4177232
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4541330, 1.4532418
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5255070, 1.5258396
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6728573, 1.6716323
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9468346, 1.9461074
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3592212, 1.3575621

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6020847, upper bound: 0.6049384
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6020847, upper bound: 0.6018465
time: 4.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2023728, 1.2020123
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4579058, 1.4584980
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5105045, 1.5105715
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395957, 1.9395580
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4177234, 1.4179082
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4532418, 1.4536359
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5256536, 1.5255067
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6716323, 1.6721699
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9461074, 1.9464211
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3575618, 1.3582895

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6018497, upper bound: 0.6051837
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6018497, upper bound: 0.6020809
time: 4.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 6, lower bound: -0.6020847, upper bound: 0.6049384
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 6, lower bound: -0.6020847, upper bound: 0.6018465
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 6, lower bound: -0.6018497, upper bound: 0.6051837
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 6, lower bound: -0.6018497, upper bound: 0.6020809

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2026043, 1.1982024
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4576001, 1.4581122
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5070980, 1.5109717
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9355578, 1.9401665
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4186902, 1.4135113
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4547555, 1.4484329
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5220339, 1.5262847
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6677651, 1.6722956
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9471684, 1.9434750
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3597622, 1.3533280

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928142, upper bound: 0.5956652
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928142, upper bound: 0.6049305
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1973934, 1.2028213
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4592495, 1.4562559
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5106609, 1.5069417
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395580, 1.9356446
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4139328, 1.4177232
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4493239, 1.4532418
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5255070, 1.5223668
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6728573, 1.6665401
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9442024, 1.9461074
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3549867, 1.3575621

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6051709, upper bound: 0.5925781
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928142, upper bound: 0.6018396
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2029648, 1.1973934
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4562559, 1.4587047
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5069420, 1.5110388
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9355960, 1.9400799
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4182687, 1.4136956
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4538643, 1.4488273
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5221808, 1.5259519
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6665401, 1.6728330
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9464412, 1.9437890
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3581033, 1.3540559

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925818, upper bound: 0.5959003
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925818, upper bound: 0.6051702
time: 4.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1977539, 1.2020123
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4579058, 1.4568484
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5105045, 1.5070088
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9395957, 1.9355581
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4135113, 1.4179082
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4484327, 1.4536359
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5256536, 1.5220339
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6716323, 1.6670778
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9434748, 1.9464211
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3533282, 1.3582895

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925818, upper bound: 0.5928143
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925818, upper bound: 0.6020716
time: 4.22 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5928142, upper bound: 0.5956652
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5928142, upper bound: 0.6049305
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.6051709, upper bound: 0.5925781
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5928142, upper bound: 0.6018396
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5925818, upper bound: 0.5959003
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5925818, upper bound: 0.6051702
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5925818, upper bound: 0.5928143
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.42
Output dim: 6, lower bound: -0.5925818, upper bound: 0.6020716

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1999938, 1.1964951
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4545393, 1.4534290
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4976037, 1.5047700
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9128881, 1.9253640
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4123316, 1.4037733
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4400952, 1.4259875
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5081782, 1.5050697
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6380172, 1.6528673
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9469652, 1.9433417
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3536134, 1.3493099

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6010236, upper bound: 0.5956645
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5931349, upper bound: 0.5956645
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2008953, 1.1955919
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4529171, 1.4550506
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5008957, 1.5014774
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9207540, 1.9174967
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4089522, 1.4071515
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4323103, 1.4337754
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5008192, 1.5124283
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6483369, 1.6425478
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9470339, 1.9432721
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3557420, 1.3471799

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5928138, upper bound: 0.5960777
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6038819
time: 8.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1947830, 1.2011139
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4561892, 1.4515727
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5011666, 1.5007393
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9168887, 1.9208407
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4075727, 1.4079847
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4346664, 1.4307971
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5116513, 1.5011518
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6431103, 1.6471121
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9439993, 1.9459741
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3488383, 1.3535442

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6041249, upper bound: 0.5925780
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5963106, upper bound: 0.5925778
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1956861, 1.2002108
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4545660, 1.4531955
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5044587, 1.4974474
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247546, 1.9129748
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4041944, 1.4113631
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4268787, 1.4385850
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5042920, 1.5085111
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6534300, 1.6367927
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9440689, 1.9459043
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3509688, 1.3514140

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5958999, upper bound: 0.5929018
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5959000, upper bound: 0.6007885
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2003546, 1.1956861
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4531956, 1.4540215
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4974473, 1.5048370
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9129262, 1.9252772
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4119101, 1.4039578
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4392040, 1.4263818
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5083253, 1.5047371
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6367927, 1.6534050
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9462380, 1.9436555
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3519549, 1.3500378

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5959000
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5929024, upper bound: 0.5959014
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.2012560, 1.1947827
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4515724, 1.4556431
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5007393, 1.5015444
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9207921, 1.9174099
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4085307, 1.4073360
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4314191, 1.4341698
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5009661, 1.5120955
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6471124, 1.6430855
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9463067, 1.9435859
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3540835, 1.3479078

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5963104
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6041270
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1951437, 1.2003050
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4548454, 1.4521651
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5010102, 1.5008063
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9169269, 1.9207540
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4071512, 1.4081697
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4337752, 1.4311914
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5117979, 1.5008192
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6418858, 1.6476498
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9432716, 1.9462876
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3471799, 1.3542717

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6038819, upper bound: 0.5928139
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5960751, upper bound: 0.5928106
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1960468, 1.1994016
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4532223, 1.4537879
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.5043023, 1.4975144
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9247928, 1.9128881
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.4037733, 1.4115481
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4259875, 1.4389794
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5044389, 1.5081782
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6522055, 1.6373301
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9433417, 1.9462180
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3493104, 1.3521414

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5931341
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6010236
time: 4.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.6010236, upper bound: 0.5956645
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5931349, upper bound: 0.5956645
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5928138, upper bound: 0.5960777
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6038819
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.6041249, upper bound: 0.5925780
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5963106, upper bound: 0.5925778
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5958999, upper bound: 0.5929018
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5959000, upper bound: 0.6007885
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5959000
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5929024, upper bound: 0.5959014
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5963104
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5925814, upper bound: 0.6041270
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.6038819, upper bound: 0.5928139
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5960751, upper bound: 0.5928106
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.70
Output dim: 6, lower bound: -0.5925814, upper bound: 0.5931341
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.70
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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5878560, upper bound: 0.5917774
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5971667, upper bound: 0.5825923
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5799544, upper bound: 0.5917790
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5892494, upper bound: 0.5825936
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5796330, upper bound: 0.5921883
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5889245, upper bound: 0.5830067
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5796330, upper bound: 0.6000246
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5889270, upper bound: 0.5908531
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5910680, upper bound: 0.5887110
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6002409, upper bound: 0.5794032
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5832286, upper bound: 0.5887115
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5924078, upper bound: 0.5794069
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5796330, upper bound: 0.5890353
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5919964, upper bound: 0.5797288
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5796330, upper bound: 0.5969525
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5889270, upper bound: 0.5876322
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5876290, upper bound: 0.5919963
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5969539, upper bound: 0.5828179
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5797281, upper bound: 0.5919976
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5890363, upper bound: 0.5828180
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5794069, upper bound: 0.5924090
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5887140, upper bound: 0.5832283
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5794069, upper bound: 0.6002415
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5887140, upper bound: 0.5910710
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5908500, upper bound: 0.5889240
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6000236, upper bound: 0.5796294
time: 5.01 seconds

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

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5830071, upper bound: 0.5889241
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5921891, upper bound: 0.5796318
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.35 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5794069, upper bound: 0.5892493
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5917776, upper bound: 0.5799544
time: 5.42 seconds

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

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5794069, upper bound: 0.5971667
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5887140, upper bound: 0.5878577
time: 5.13 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5878560, upper bound: 0.5917774
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5971667, upper bound: 0.5825923
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5799544, upper bound: 0.5917790
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5892494, upper bound: 0.5825936
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5796330, upper bound: 0.5921883
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5889245, upper bound: 0.5830067
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5796330, upper bound: 0.6000246
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5889270, upper bound: 0.5908531
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5910680, upper bound: 0.5887110
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.6002409, upper bound: 0.5794032
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5832286, upper bound: 0.5887115
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5924078, upper bound: 0.5794069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5796330, upper bound: 0.5890353
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5919964, upper bound: 0.5797288
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5796330, upper bound: 0.5969525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5889270, upper bound: 0.5876322
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5876290, upper bound: 0.5919963
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5969539, upper bound: 0.5828179
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5797281, upper bound: 0.5919976
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5890363, upper bound: 0.5828180
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5794069, upper bound: 0.5924090
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5887140, upper bound: 0.5832283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5794069, upper bound: 0.6002415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5887140, upper bound: 0.5910710
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5908500, upper bound: 0.5889240
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.6000236, upper bound: 0.5796294
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5830071, upper bound: 0.5889241
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5921891, upper bound: 0.5796318
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5794069, upper bound: 0.5892493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5917776, upper bound: 0.5799544
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5794069, upper bound: 0.5971667
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.02
Output dim: 6, lower bound: -0.5887140, upper bound: 0.5878577

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1997962, 1.1962476
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4540989, 1.4530504
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4844553, 1.4912605
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.9105339, 1.9245503
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3836765, 1.3709624
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4228466, 1.4093108
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.5038383, 1.5024714
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6263678, 1.6431618
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.9291611, 1.9297109
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3531075, 1.3482850

Time for backsubstitution: 14.78 seconds
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

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4477134
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4479308
time: 4.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.18
Output dim: 6, lower bound: -0.4477160, upper bound: 0.4477134
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.18
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

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466573, upper bound: 0.4477110
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466572, upper bound: 0.4466554
time: 4.59 seconds

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

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466573, upper bound: 0.4479276
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4466572, upper bound: 0.4466553
time: 7.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.96
Output dim: 6, lower bound: -0.4466573, upper bound: 0.4477110
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.96
Output dim: 6, lower bound: -0.4466572, upper bound: 0.4466554
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.96
Output dim: 6, lower bound: -0.4466573, upper bound: 0.4479276
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.96
Output dim: 6, lower bound: -0.4466572, upper bound: 0.4466553

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

Time for backsubstitution: 15.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4445080
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4477111
time: 4.80 seconds

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

Time for backsubstitution: 15.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4434750
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4466535
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0981069, 1.0961003
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3424482, 1.3435152
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4164555, 1.4178507
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7839761, 1.7854545
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3288884, 1.3274431
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3613679, 1.3598545
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4483151, 1.4495101
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5409908, 1.5433176
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8369083, 1.8361623
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2856798, 1.2846410

Time for backsubstitution: 15.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4447262
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4479258
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0963700, 1.0978372
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3430672, 1.3428962
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4177988, 1.4165074
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7854834, 1.7839472
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3273029, 1.3290291
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3595574, 1.3616650
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4496212, 1.4482043
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5429091, 1.5413990
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8359199, 1.8371511
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2840881, 1.2862327

Time for backsubstitution: 17.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4436931
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4468682
time: 4.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4445080
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4477111
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4434750
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4466535
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4447262
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4479258
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.40
Output dim: 6, lower bound: -0.4434770, upper bound: 0.4436931
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.40
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

Time for backsubstitution: 18.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4445071
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4445063
time: 5.72 seconds

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

Time for backsubstitution: 17.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4446534
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4477079
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0937908, 1.0954964
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3388319, 1.3383060
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4094541, 1.4069610
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7654071, 1.7613063
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3177052, 1.3202770
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3374088, 1.3415174
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4282954, 1.4295533
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5170097, 1.5112431
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8359823, 1.8367057
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2792029, 1.2795315

Time for backsubstitution: 15.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434764, upper bound: 0.4435855
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4466499
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0954964, 1.0937908
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3383064, 1.3388319
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4069612, 1.4094539
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7613063, 1.7654071
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3202772, 1.3177052
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3415172, 1.3374090
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4295533, 1.4282954
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5112433, 1.5170097
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8367052, 1.8359826
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2795320, 1.2792029

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4447244
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4435857, upper bound: 0.4447228
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4448724
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4479254
time: 5.04 seconds

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

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4438026
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4468670
time: 4.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4445071
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4445063
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4446534
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4477079
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434764, upper bound: 0.4435855
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4466499
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4447244
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4435857, upper bound: 0.4447228
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4448724
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4479254
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.50
Output dim: 6, lower bound: -0.4434765, upper bound: 0.4438026
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
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

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4436566
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4400328
time: 4.80 seconds

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

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4393259, upper bound: 0.4436554
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4429550, upper bound: 0.4400288
time: 5.12 seconds

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

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4438033
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4401750
time: 4.55 seconds

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

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4468592
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4432357
time: 4.34 seconds

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

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4426257
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4421740
time: 4.74 seconds

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

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4438707
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4402473
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4391113, upper bound: 0.4438725
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4427403, upper bound: 0.4402434
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4440179
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4403941
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4470746
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4434458
time: 4.52 seconds

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

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4460179
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4423886
time: 4.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4436566
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4400328
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4393259, upper bound: 0.4436554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4429550, upper bound: 0.4400288
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4438033
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4401750
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4468592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4432357
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4426257
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4421740
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4438707
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4402473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4391113, upper bound: 0.4438725
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4427403, upper bound: 0.4402434
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4440179
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4403941
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4470746
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4434458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390021, upper bound: 0.4460179
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.60
Output dim: 6, lower bound: -0.4390022, upper bound: 0.4423886

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0952961, 1.0935111
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3377678, 1.3385029
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3949611, 1.3950348
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7615461, 1.7609715
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2906303, 1.2886486
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3223503, 1.3230281
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4226437, 1.4270978
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5047357, 1.5034509
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8219490, 1.8220859
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2799401, 1.2769101

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357318, upper bound: 0.4436890
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4435921
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0952649, 1.0935428
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3378623, 1.3384087
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3938119, 1.3961844
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7589521, 1.7635665
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2916183, 1.2876611
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3246477, 1.3207302
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4252095, 1.4245322
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5008862, 1.5073006
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8216839, 1.8223517
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2786765, 1.2781746

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4407004
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4406010
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0952647, 1.0935423
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3378608, 1.3384097
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3938115, 1.3961840
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7589531, 1.7635651
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2916164, 1.2876627
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3246467, 1.3207293
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4252076, 1.4245322
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5008872, 1.5072989
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8216839, 1.8223515
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2786765, 1.2781732

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4358407, upper bound: 0.4407017
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4359532, upper bound: 0.4406048
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0955656, 1.0932410
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3373210, 1.3389492
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3949082, 1.3950869
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7615738, 1.7609439
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2904921, 1.2887871
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3220532, 1.3233240
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4227552, 1.4269850
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5043261, 1.5038607
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8217063, 1.8223286
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2793860, 1.2774637

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357318, upper bound: 0.4408492
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4407485
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0955658, 1.0932412
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3373196, 1.3389509
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3949091, 1.3950869
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7615752, 1.7609427
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2904902, 1.2887890
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3220537, 1.3233252
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4227545, 1.4269869
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5043275, 1.5038590
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8217068, 1.8223286
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2793875, 1.2774632

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.4357318, upper bound: 0.4439040
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4438049
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.0938294, 1.0949783
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3379385, 1.3383324
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3962524, 1.3937440
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.7630830, 1.7594354
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2889042, 1.2903752
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3202436, 1.3251345
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4240606, 1.4256811
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5062463, 1.5019407
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8207183, 1.8233171
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2777963, 1.2790549

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1459

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4428435
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4427539
time: 4.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.60 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357318, upper bound: 0.4436890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4435921
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4407004
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4406010
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4358407, upper bound: 0.4407017
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4359532, upper bound: 0.4406048
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357318, upper bound: 0.4408492
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4407485
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357318, upper bound: 0.4439040
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4438049
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4428435
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 23.60
Output dim: 6, lower bound: -0.4357317, upper bound: 0.4427539

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1026721, 1.0999308
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3109331, 1.3130369
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.3552601, 1.3590767
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.6737571, 1.6719604
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.2557826, 1.2533174
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3344440, 1.3342927
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4235308, 1.4317300
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.4816127, 1.4789534
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8381824, 1.8372939
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.2620213, 1.2594910

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Candidate
type: RSZ, layer: 3, pos: 1704

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4309405, upper bound: 0.4432672
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.4351915, upper bound: 0.4390123
time: 4.95 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.54 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.54
Output dim: 6, lower bound: -0.4309405, upper bound: 0.4432672
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.54
Output dim: 6, lower bound: -0.4351915, upper bound: 0.4390123
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.4518237113952637
rel_dist={6: [-0.44793373691976335, 0.4479324567017895]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4626
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4626

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5284015, upper bound: 0.5283989
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5284015, upper bound: 0.5286440
time: 4.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.91
Output dim: 6, lower bound: -0.5284015, upper bound: 0.5283989
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.91
Output dim: 6, lower bound: -0.5284015, upper bound: 0.5286440

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1513658, 1.1519051
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4018979, 1.4010017
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4653656, 1.4652611
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8637528, 1.8638105
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3749001, 1.3746190
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4093981, 1.4088042
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4885921, 1.4888136
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6096740, 1.6088576
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8928146, 1.8923297
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3240478, 1.3229423

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5283940
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5260941
time: 4.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1517262, 1.1513658
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4010019, 1.4015942
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4652612, 1.4653282
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8637905, 1.8637526
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3746188, 1.3748040
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4088039, 1.4091983
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4887388, 1.4885919
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6088576, 1.6093953
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8923297, 1.8926435
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3229425, 1.3236697

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6114
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6114

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5286387
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5263400
time: 4.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.30
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5283940
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.30
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5260941
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.30
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5286387
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.30
Output dim: 6, lower bound: -0.5260968, upper bound: 0.5263400

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1502209, 1.1472862
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4002481, 1.4005897
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4618027, 1.4643852
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8597527, 1.8628249
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3738599, 1.3704071
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4082100, 1.4039950
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4851191, 1.4879527
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6045818, 1.6076024
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8921595, 1.8896976
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3229976, 1.3187082

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5221016
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5283874
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1467469, 1.1507602
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4014859, 1.3993523
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4644892, 1.4616984
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8627672, 1.8598104
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3706880, 1.3735788
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4045889, 1.4076161
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4877310, 1.4853408
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6084189, 1.6037655
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8901825, 1.8916750
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3198137, 1.3218915

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5198161
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5260870
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1505814, 1.1467469
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3993526, 1.4011822
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4616988, 1.4644523
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8597908, 1.8627672
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3735785, 1.3705912
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4076159, 1.4043896
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4852660, 1.4877310
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6037655, 1.6081400
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8916750, 1.8900113
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3218918, 1.3194358

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5223466
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5286326
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1471074, 1.1502209
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.4005899, 1.3999445
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4643853, 1.4617655
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8628054, 1.8597527
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3704071, 1.3737631
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.4039948, 1.4080107
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4878781, 1.4851191
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.6076026, 1.6043029
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8896976, 1.8919888
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3187079, 1.3226192

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 430
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5200578
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5263321
time: 4.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5221016
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5283874
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5198161
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5260870
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5223466
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5286326
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5200578
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.00
Output dim: 6, lower bound: -0.5198161, upper bound: 0.5263321

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1476102, 1.1452777
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3966465, 1.3959064
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4523084, 1.4570860
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8370829, 1.8453999
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3663745, 1.3606689
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3909547, 1.3815498
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4688103, 1.4667380
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5748343, 1.5847344
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8919563, 1.8895409
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3168492, 1.3139799

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5263331, upper bound: 0.5221002
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5202748, upper bound: 0.5221000
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1482112, 1.1446755
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3955650, 1.3969874
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4545033, 1.4548908
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8423266, 1.8401551
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3641214, 1.3629212
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3857648, 1.3867416
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4639044, 1.4716437
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5817142, 1.5778549
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8920021, 1.8894944
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3182683, 1.3125598

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5223932
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5283873
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1441364, 1.1487505
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3978834, 1.3946691
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4549954, 1.4543988
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8400974, 1.8423846
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3632021, 1.3638406
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3873355, 1.3851709
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4714220, 1.4641261
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5786715, 1.5808976
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8899794, 1.8915176
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3136659, 1.3171623

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5198130
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5226371, upper bound: 0.5198137
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1447384, 1.1481495
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3968029, 1.3957508
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4571903, 1.4522041
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8453422, 1.8371406
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3609500, 1.3660936
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3821437, 1.3903606
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4665163, 1.4690323
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5855508, 1.5740180
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8900256, 1.8914719
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3150859, 1.3157434

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5200320
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5260861
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1479709, 1.1447384
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3957510, 1.3964989
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4522040, 1.4571530
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8371210, 1.8453422
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3660936, 1.3608534
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3903606, 1.3819442
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4689574, 1.4665163
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5740180, 1.5852718
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8914719, 1.8898547
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3157430, 1.3147078

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5223460
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5200339, upper bound: 0.5223450
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1485720, 1.1441362
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3946686, 1.3975799
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4543989, 1.4549578
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8423648, 1.8400974
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3638406, 1.3631058
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3851707, 1.3871360
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4640512, 1.4714220
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5808978, 1.5783923
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8915176, 1.8898082
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3171620, 1.3132877

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5226374
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5286325
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1444972, 1.1482112
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3969879, 1.3952613
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4548910, 1.4544657
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8401356, 1.8423266
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3629212, 1.3640251
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3867414, 1.3855653
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4715688, 1.4639044
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5778546, 1.5814352
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8894944, 1.8918314
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3125596, 1.3178899

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5200542
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5223935, upper bound: 0.5200540
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1450992, 1.1476102
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3959064, 1.3963432
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4570858, 1.4522711
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8453803, 1.8370829
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3606691, 1.3662782
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3815496, 1.3907552
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4666634, 1.4688103
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5847344, 1.5745554
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8895407, 1.8917856
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3139796, 1.3164711

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5202743
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5263360
time: 4.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5263331, upper bound: 0.5221002
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5202748, upper bound: 0.5221000
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5223932
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5283873
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5198130
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5226371, upper bound: 0.5198137
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5200320
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5260861
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5223460
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5200339, upper bound: 0.5223450
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5226374
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5286325
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5200542
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5223935, upper bound: 0.5200540
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5202743
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 6, lower bound: -0.5198157, upper bound: 0.5263360

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1476095, 1.1452777
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3966458, 1.3959024
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4523079, 1.4570848
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8370795, 1.8453994
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3663731, 1.3606639
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3909514, 1.3815486
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4688101, 1.4667339
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5748296, 1.5847332
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8919568, 1.8895407
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3168459, 1.3139794

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5194744
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5237485, upper bound: 0.5130443
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1476088, 1.1452770
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3966424, 1.3959041
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4523070, 1.4570838
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8370814, 1.8453970
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3663692, 1.3606677
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3909495, 1.3815465
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4688063, 1.4667342
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5748320, 1.5847297
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8919568, 1.8895407
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3168459, 1.3139763

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5112388, upper bound: 0.5194735
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5176684, upper bound: 0.5130448
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1482105, 1.1446743
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3955629, 1.3969834
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4545014, 1.4548897
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8423238, 1.8401546
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3641205, 1.3629162
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3857615, 1.3867364
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4639015, 1.4716396
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5817094, 1.5778537
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8920021, 1.8894942
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3182645, 1.3125579

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5197646
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5133335
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1482110, 1.1446748
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3955610, 1.3969867
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4545023, 1.4548897
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8423262, 1.8401520
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3641167, 1.3629198
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3857625, 1.3867385
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4639001, 1.4716432
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5817132, 1.5778501
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8920026, 1.8894944
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3182678, 1.3125563

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5258058
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5193704
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1441355, 1.1487505
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3978827, 1.3946650
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4549944, 1.4543976
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8400941, 1.8423839
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3632011, 1.3638358
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3873327, 1.3851686
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4714215, 1.4641218
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5786667, 1.5808964
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8899794, 1.8915172
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3136621, 1.3171618

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5172045
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5260499, upper bound: 0.5107707
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1441350, 1.1487498
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3978794, 1.3946669
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4549935, 1.4543967
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8400970, 1.8423815
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3631973, 1.3638396
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3873303, 1.3851676
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4714177, 1.4641232
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5786700, 1.5808930
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8899789, 1.8915172
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3136640, 1.3171587

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5135770, upper bound: 0.5172027
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5200095, upper bound: 0.5107713
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1447377, 1.1481481
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3967998, 1.3957467
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4571879, 1.4522029
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8453393, 1.8371391
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3609486, 1.3660886
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3821404, 1.3903553
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4665124, 1.4690280
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5855465, 1.5740151
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8900256, 1.8914716
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3150821, 1.3157399

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5174251
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172067, upper bound: 0.5109912
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1447382, 1.1481488
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3967984, 1.3957500
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4571888, 1.4522039
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8453417, 1.8371375
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3609447, 1.3660922
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3821428, 1.3903575
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4665120, 1.4690318
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5855498, 1.5740132
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8900256, 1.8914716
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3150854, 1.3157399

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5106798
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5170692
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1479702, 1.1447382
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3957498, 1.3964946
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4522040, 1.4571518
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8371181, 1.8453417
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3660927, 1.3608484
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3903573, 1.3819430
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4689569, 1.4665122
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5740132, 1.5852709
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8914719, 1.8898551
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3157401, 1.3147073

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5197187
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5235046, upper bound: 0.5132878
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1479697, 1.1447377
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3957465, 1.3964961
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4522030, 1.4571508
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8371201, 1.8453391
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3660889, 1.3608522
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3903553, 1.3819408
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4689531, 1.4665124
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5740151, 1.5852675
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8914719, 1.8898549
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3157401, 1.3147042

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5109921, upper bound: 0.5197174
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5174245, upper bound: 0.5132860
time: 4.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1485713, 1.1441348
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3946669, 1.3975756
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4543965, 1.4549567
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8423619, 1.8400967
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3638391, 1.3631008
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3851674, 1.3871307
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4640484, 1.4714177
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5808930, 1.5783911
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8915172, 1.8898087
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3171587, 1.3132856

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5200096
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172070, upper bound: 0.5135769
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1485720, 1.1441355
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3946650, 1.3975787
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4543974, 1.4549567
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8423648, 1.8400943
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3638353, 1.3631043
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3851683, 1.3871329
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4640472, 1.4714215
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5808964, 1.5783877
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8915172, 1.8898087
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3171620, 1.3132842

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5260488
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5196153
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1444964, 1.1482110
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3969867, 1.3952570
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4548895, 1.4544646
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8401327, 1.8423262
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3629198, 1.3640203
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3867385, 1.3855629
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4715683, 1.4639001
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5778503, 1.5814340
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8894944, 1.8918319
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3125563, 1.3178895

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5174484
time: 11.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5258061, upper bound: 0.5110176
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1444957, 1.1482105
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3969834, 1.3952591
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4548895, 1.4544637
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8401351, 1.8423238
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3629160, 1.3640242
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3867362, 1.3855622
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4715648, 1.4639015
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5778537, 1.5814307
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8894944, 1.8918319
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3125577, 1.3178864

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5133355, upper bound: 0.5174459
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5197650, upper bound: 0.5110175
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1450984, 1.1476088
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3959038, 1.3963387
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4570839, 1.4522699
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8453774, 1.8370814
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3606672, 1.3662732
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3815463, 1.3907497
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4666593, 1.4688063
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5847297, 1.5745530
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8895407, 1.8917861
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3139763, 1.3164678

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5176669
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172070, upper bound: 0.5112379
time: 5.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -4.8109689, -3.0668516, -4.8109689, -3.0668516, -1.1450992, 1.1476095
1: -2.8668113, -0.8982847, -2.8668113, -0.8982847, -1.3959024, 1.3963420
2: -4.2153616, -2.3826535, -4.2153616, -2.3826535, -1.4570849, 1.4522709
3: -12.6510830, -9.9383640, -12.6510830, -9.9383640, -1.8453803, 1.8370798
4: -6.0474753, -4.2851191, -6.0474753, -4.2851191, -1.3606644, 1.3662767
5: -2.8278728, -1.0084610, -2.8278728, -1.0084610, -1.3815486, 1.3907518
6: 2.2630239, 3.8312683, 2.2630239, 3.8312683, -1.4666591, 1.4688101
7: -10.2777958, -8.1927624, -10.2777958, -8.1927624, -1.5847335, 1.5745509
8: -1.9165163, 0.7287664, -1.9165163, 0.7287664, -1.8895407, 1.8917861
9: -8.5035858, -6.9852848, -8.5035858, -6.9852848, -1.3139796, 1.3164675

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2810
type: RSZ, layer: 3, pos: 1459
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 1697
type: RSZ, layer: 3, pos: 1745
type: RSZ, layer: 3, pos: 1116
type: RSZ, layer: 3, pos: 2326
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1389
type: RSZ, layer: 3, pos: 422
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 773
type: RSZ, layer: 3, pos: 563
type: RSZ, layer: 3, pos: 1824
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 710
type: RSZ, layer: 3, pos: 213
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1249
type: RSZ, layer: 3, pos: 1837
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1696
type: RSZ, layer: 3, pos: 2804
type: RSZ, layer: 3, pos: 1242
type: RSZ, layer: 3, pos: 1263
type: RSZ, layer: 3, pos: 1411
type: RSZ, layer: 3, pos: 2467
type: RSZ, layer: 3, pos: 2342
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 1780
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 2333
type: RSZ, layer: 3, pos: 2816
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2495
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 2634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 1789
type: RSZ, layer: 3, pos: 2588
type: RSZ, layer: 3, pos: 2902
type: RSZ, layer: 3, pos: 2930
type: RSZ, layer: 3, pos: 1727
type: RSZ, layer: 3, pos: 2455
type: RSZ, layer: 3, pos: 1769
type: RSZ, layer: 3, pos: 2307

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 2810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5237475
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5173175
time: 4.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5194744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5237485, upper bound: 0.5130443
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5112388, upper bound: 0.5194735
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5176684, upper bound: 0.5130448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5197646
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5133335
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5258058
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5193704
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5172045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5260499, upper bound: 0.5107707
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5135770, upper bound: 0.5172027
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5200095, upper bound: 0.5107713
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5174251
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172067, upper bound: 0.5109912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5106798
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5170692
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5197187
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5235046, upper bound: 0.5132878
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5109921, upper bound: 0.5197174
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5174245, upper bound: 0.5132860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5200096
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172070, upper bound: 0.5135769
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5260488
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5196153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5174484
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5258061, upper bound: 0.5110176
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5133355, upper bound: 0.5174459
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5197650, upper bound: 0.5110175
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5176669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172070, upper bound: 0.5112379
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5107748, upper bound: 0.5237475
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.90
Output dim: 6, lower bound: -0.5172069, upper bound: 0.5173175
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.488738775253296
rel_dist={6: [-0.5286543609646217, 0.528651538561538]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 3208.11 seconds
