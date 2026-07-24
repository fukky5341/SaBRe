## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.61680046168
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.5487733, 4.5487733)
1: (-17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203)
2: (-8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.9031115, 3.9031115)
3: (-13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582)
4: (-3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6634710, 3.6634710)
5: (-13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.8109388, 3.8109393)
6: (-15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.5555229, 4.5555229)
7: (-8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024)
8: (-6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597)
9: (3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937)

## BASE Result
execution time: IAR + LP analysis = 15.20 + 33.98 = 49.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.0738973, upper bound: 2.0738940


# Binary Search by BASE starts (time budget: 3550.83 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.6592936515808105
rel_dist={9: [-1.6176245637311917, 1.6176249981850015]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.6068572998046875
rel_dist={9: [-1.2926447051799563, 1.2926467080621613]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.6592936515808105
rel_dist={9: [-1.417875488335799, 1.4178760636054193]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=2.6592936515808105
rel_dist={9: [-1.5239616905627367, 1.523961546591452]}

## Binary Search Result
Binary search time: 205.78 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 3345.05 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8603862, upper bound: 1.8341501
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8341504, upper bound: 1.8603867
time: 4.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.08 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.08
Output dim: 9, lower bound: -1.8603862, upper bound: 1.8341501
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.08
Output dim: 9, lower bound: -1.8341504, upper bound: 1.8603867

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1660995, 4.1685162
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6191368, 3.6322432
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6025424, 3.6147003
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3800373, 3.3758774
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2416239, 4.2285852
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8592210, upper bound: 1.8341505
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8603826, upper bound: 1.8329847
time: 4.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1685162, 4.1661000
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6322432, 3.6191359
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6147008, 3.6025419
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3758774, 3.3800373
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2285852, 4.2416239
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8329824, upper bound: 1.8603829
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8341502, upper bound: 1.8592217
time: 5.10 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.98
Output dim: 9, lower bound: -1.8592210, upper bound: 1.8341505
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.98
Output dim: 9, lower bound: -1.8603826, upper bound: 1.8329847
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.98
Output dim: 9, lower bound: -1.8329824, upper bound: 1.8603829
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.98
Output dim: 9, lower bound: -1.8341502, upper bound: 1.8592217

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1604795, 4.1505790
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5964918, 3.6180863
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5925746, 3.5942774
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3669157, 3.3549104
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2552032, 4.2383604
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8592180, upper bound: 1.8275438
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8525489, upper bound: 1.8341475
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1481628, 4.1628952
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6049795, 3.6095982
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5821195, 3.6047325
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3590708, 3.3627558
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2513981, 4.2421646
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8603796, upper bound: 1.8263780
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8537107, upper bound: 1.8329794
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1628952, 4.1481628
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6095991, 3.6049795
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.6047330, 3.5821195
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3627558, 3.3590703
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2421646, 4.2513986
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8329794, upper bound: 1.8537110
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8263781, upper bound: 1.8603804
time: 7.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1505795, 4.1604795
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6180868, 3.5964913
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5942779, 3.5925751
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3549109, 3.3669162
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2383604, 4.2552032
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8341472, upper bound: 1.8525495
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8275416, upper bound: 1.8592183
time: 5.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8592180, upper bound: 1.8275438
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8525489, upper bound: 1.8341475
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8603796, upper bound: 1.8263780
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8537107, upper bound: 1.8329794
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8329794, upper bound: 1.8537110
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8263781, upper bound: 1.8603804
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8341472, upper bound: 1.8525495
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 9, lower bound: -1.8275416, upper bound: 1.8592183

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1572180, 4.1453729
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5866585, 3.6119337
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5851521, 3.5823998
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3607540, 3.3450584
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2523079, 4.2365518
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8592123, upper bound: 1.7999841
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8316378, upper bound: 1.8275353
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1552734, 4.1473179
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5903387, 3.6082535
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5806975, 3.5868549
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3570642, 3.3487482
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2533951, 4.2354641
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8525431, upper bound: 1.8065921
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8249687, upper bound: 1.8341423
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1449022, 4.1576891
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5951471, 3.6034455
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5746980, 3.5928555
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3529081, 3.3529038
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2485027, 4.2403560
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8603739, upper bound: 1.7988201
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8327995, upper bound: 1.8263716
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1429567, 4.1596346
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5988274, 3.5997648
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5702424, 3.5973105
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3492184, 3.3565941
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2495899, 4.2392688
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8537050, upper bound: 1.8054244
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8261306, upper bound: 1.8329744
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1596346, 4.1429567
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5997648, 3.5988269
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5973105, 3.5702419
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3565941, 3.3492184
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2392693, 4.2495899
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8329737, upper bound: 1.8261310
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8054245, upper bound: 1.8537055
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1576891, 4.1449018
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6034460, 3.5951467
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5928559, 3.5746975
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3529043, 3.3529086
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2403564, 4.2485027
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8263724, upper bound: 1.8328001
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7988204, upper bound: 1.8603741
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1473179, 4.1552734
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6082535, 3.5903387
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5868554, 3.5806975
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3487482, 3.3570642
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2354641, 4.2533946
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8341415, upper bound: 1.8249694
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065922, upper bound: 1.8525433
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1453733, 4.1572185
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6119337, 3.5866580
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5823998, 3.5851526
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3450584, 3.3607540
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2365513, 4.2523074
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8275359, upper bound: 1.8316384
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7999837, upper bound: 1.8592129
time: 4.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8592123, upper bound: 1.7999841
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8316378, upper bound: 1.8275353
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8525431, upper bound: 1.8065921
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8249687, upper bound: 1.8341423
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8603739, upper bound: 1.7988201
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8327995, upper bound: 1.8263716
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8537050, upper bound: 1.8054244
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8261306, upper bound: 1.8329744
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8329737, upper bound: 1.8261310
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8054245, upper bound: 1.8537055
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8263724, upper bound: 1.8328001
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.7988204, upper bound: 1.8603741
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8341415, upper bound: 1.8249694
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8065922, upper bound: 1.8525433
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.8275359, upper bound: 1.8316384
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.88
Output dim: 9, lower bound: -1.7999837, upper bound: 1.8592129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1489019, 4.1401744
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5918274, 3.6151671
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5789990, 3.5725532
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3524513, 3.3398666
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2445726, 4.2317181
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8592097, upper bound: 1.7997144
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8523230, upper bound: 1.7997189
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1520205, 4.1370564
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5898914, 3.6171026
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5753064, 3.5762467
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3555632, 3.3367548
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2474737, 4.2288170
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8316352, upper bound: 1.8272659
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8247486, upper bound: 1.8272696
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1469564, 4.1421194
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5955076, 3.6114864
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5745435, 3.5770082
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3487606, 3.3435569
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2456598, 4.2306309
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8523230, upper bound: 1.7997190
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8523185, upper bound: 1.8065890
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1500750, 4.1390018
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5935717, 3.6134224
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5708508, 3.5807023
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3518724, 3.3404450
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2485609, 4.2277298
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8247485, upper bound: 1.8272700
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8247440, upper bound: 1.8341376
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1365862, 4.1524911
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6003151, 3.6066785
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5685449, 3.5830083
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3446054, 3.3477125
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2407684, 4.2355223
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8603713, upper bound: 1.7985471
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8534847, upper bound: 1.7985516
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1397038, 4.1493731
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5983801, 3.6086144
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5648503, 3.5867023
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3477173, 3.3446007
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2436695, 4.2326212
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8327969, upper bound: 1.8260961
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8259103, upper bound: 1.8261006
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1346407, 4.1544361
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6039963, 3.6029983
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5640893, 3.5874639
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3409147, 3.3514028
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2418556, 4.2344351
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8534846, upper bound: 1.7985514
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8534801, upper bound: 1.8054210
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1377583, 4.1513181
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6020603, 3.6049337
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5603948, 3.5911574
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3440266, 3.3482909
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2447567, 4.2315340
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8259102, upper bound: 1.8261013
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8259057, upper bound: 1.8329693
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1513186, 4.1377583
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6049337, 3.6020603
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5911565, 3.5603952
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3482904, 3.3440270
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2315340, 4.2447567
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8329701, upper bound: 1.8259064
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8261014, upper bound: 1.8259104
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1544361, 4.1346407
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6029987, 3.6039963
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5874639, 3.5640888
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3514023, 3.3409147
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2344351, 4.2418556
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8054211, upper bound: 1.8534806
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7985519, upper bound: 1.8534868
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1493731, 4.1397038
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6086140, 3.5983796
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5867028, 3.5648503
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3446007, 3.3477168
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2326212, 4.2436690
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8261013, upper bound: 1.8259104
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8260969, upper bound: 1.8327975
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1524916, 4.1365857
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6066790, 3.6003156
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5830083, 3.5685444
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3477125, 3.3446050
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2355223, 4.2407680
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7985518, upper bound: 1.8534848
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7985474, upper bound: 1.8603726
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1390018, 4.1500750
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6134224, 3.5935717
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5807023, 3.5708504
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3404455, 3.3518724
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2277298, 4.2485609
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8341381, upper bound: 1.8247445
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8272701, upper bound: 1.8247488
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1421194, 4.1469569
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6114864, 3.5955076
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5770078, 3.5745444
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3435574, 3.3487606
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2306309, 4.2456598
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8065888, upper bound: 1.8523192
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7997191, upper bound: 1.8523231
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1370564, 4.1520205
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6171036, 3.5898914
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5762467, 3.5753059
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3367548, 3.3555627
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2288170, 4.2474737
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8272700, upper bound: 1.8247491
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8272656, upper bound: 1.8316355
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1401739, 4.1489019
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.6151676, 3.5918269
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5725522, 3.5789995
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3398666, 3.3524508
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2317181, 4.2445726
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7997190, upper bound: 1.8523232
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7997145, upper bound: 1.8592097
time: 4.95 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8592097, upper bound: 1.7997144
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8523230, upper bound: 1.7997189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8316352, upper bound: 1.8272659
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8247486, upper bound: 1.8272696
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8523230, upper bound: 1.7997190
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8523185, upper bound: 1.8065890
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8247485, upper bound: 1.8272700
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8247440, upper bound: 1.8341376
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8603713, upper bound: 1.7985471
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8534847, upper bound: 1.7985516
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8327969, upper bound: 1.8260961
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8259103, upper bound: 1.8261006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8534846, upper bound: 1.7985514
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8534801, upper bound: 1.8054210
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8259102, upper bound: 1.8261013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8259057, upper bound: 1.8329693
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8329701, upper bound: 1.8259064
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8261014, upper bound: 1.8259104
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8054211, upper bound: 1.8534806
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.7985519, upper bound: 1.8534868
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8261013, upper bound: 1.8259104
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8260969, upper bound: 1.8327975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.7985518, upper bound: 1.8534848
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.7985474, upper bound: 1.8603726
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8341381, upper bound: 1.8247445
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8272701, upper bound: 1.8247488
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8065888, upper bound: 1.8523192
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.7997191, upper bound: 1.8523231
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8272700, upper bound: 1.8247491
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.8272656, upper bound: 1.8316355
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.7997190, upper bound: 1.8523232
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.43
Output dim: 9, lower bound: -1.7997145, upper bound: 1.8592097

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -4.1429710, 4.1306992
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.7881203, 3.7881203
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.5531530, 3.6009078
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.5645523, 3.5494437
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.3508291, 3.3372731
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -4.2343035, 4.2267971
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.2089024, 4.2089024
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.6592936515808105
rel_dist={9: [-1.860453671254131, 1.860453011622619]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023046, upper bound: 1.6809389
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6809387, upper bound: 1.7023045
time: 5.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.76
Output dim: 9, lower bound: -1.7023046, upper bound: 1.6809389
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.76
Output dim: 9, lower bound: -1.6809387, upper bound: 1.7023045

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8847904, 3.8866701
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6307592, 3.6313872
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4276714, 3.4378653
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3712144, 3.3806705
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0964632, 3.0932274
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9777679, 3.9676270
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1925421, 4.1888285
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014880, upper bound: 1.6809358
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7023021, upper bound: 1.6801393
time: 4.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8866701, 3.8847909
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6313868, 3.6307592
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4378662, 3.4276714
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3806710, 3.3712144
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0932274, 3.0964632
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9676266, 3.9777679
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1888285, 4.1925421
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6801390, upper bound: 1.7023026
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6809363, upper bound: 1.7014883
time: 4.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.7014880, upper bound: 1.6809358
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.7023021, upper bound: 1.6801393
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.6801390, upper bound: 1.7023026
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.6809363, upper bound: 1.7014883

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8764334, 3.8687329
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6522980, 3.6472311
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4050264, 3.4218230
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3589234, 3.3602476
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0815992, 3.0722609
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9905014, 3.9774017
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1911917, 4.1950312
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014857, upper bound: 1.6759101
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6964965, upper bound: 1.6809341
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8668537, 3.8783121
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6466026, 3.6529264
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4116297, 3.4152207
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3507915, 3.3683796
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0754967, 3.0783629
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9875431, 3.9803610
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1987448, 4.1874781
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7022998, upper bound: 1.6751156
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6973140, upper bound: 1.6801366
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8783121, 3.8668537
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6529264, 3.6466031
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4152212, 3.4116287
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3683791, 3.3507915
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0783634, 3.0754962
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9803610, 3.9875431
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1874781, 4.1987448
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6801366, upper bound: 1.6973141
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6751164, upper bound: 1.7022996
time: 5.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8687325, 3.8764334
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6472311, 3.6522980
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4218225, 3.4050264
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3602481, 3.3589234
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0722609, 3.0815983
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9774017, 3.9905019
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1950312, 4.1911917
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6809339, upper bound: 1.6964963
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6759101, upper bound: 1.7014857
time: 5.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.7014857, upper bound: 1.6759101
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.6964965, upper bound: 1.6809341
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.7022998, upper bound: 1.6751156
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.6973140, upper bound: 1.6801366
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.6801366, upper bound: 1.6973141
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.6751164, upper bound: 1.7022996
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.6809339, upper bound: 1.6964963
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 9, lower bound: -1.6759101, upper bound: 1.7014857

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8727398, 3.8635268
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6486511, 3.6446495
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3951941, 3.4148521
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3505111, 3.3483706
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0746164, 3.0624084
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9876060, 3.9753518
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1836462, 4.1843824
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014813, upper bound: 1.6544474
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6800202, upper bound: 1.6759049
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8712273, 3.8650398
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6497164, 3.6435847
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3980560, 3.4119897
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3470464, 3.3518353
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0717468, 3.0652790
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9884520, 3.9745059
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1805429, 4.1874847
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6964921, upper bound: 1.6594716
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6750284, upper bound: 1.6809294
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8631601, 3.8731060
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6429567, 3.6503439
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4017954, 3.4082503
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3423791, 3.3565025
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0685148, 3.0685110
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9846478, 3.9783106
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1911993, 4.1768293
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7022954, upper bound: 1.6536540
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6808337, upper bound: 1.6751119
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8616476, 3.8746190
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6440210, 3.6492796
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4046583, 3.4053874
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3389144, 3.3599672
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0656443, 3.0713811
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9854927, 3.9774652
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1880960, 4.1799316
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6973096, upper bound: 1.6586778
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6758453, upper bound: 1.6801345
time: 8.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8746195, 3.8616476
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6492796, 3.6440210
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4053879, 3.4046583
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3599677, 3.3389139
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0713806, 3.0656443
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9774647, 3.9854927
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1799326, 4.1880960
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6801322, upper bound: 1.6758456
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6586755, upper bound: 1.6973098
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8731060, 3.8631606
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6503439, 3.6429567
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4082499, 3.4017954
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3565021, 3.3423796
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0685110, 3.0685143
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9783106, 3.9846473
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1768293, 4.1911983
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6751120, upper bound: 1.6808335
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6536537, upper bound: 1.7022951
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8650389, 3.8712273
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6435852, 3.6497164
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4119902, 3.3980560
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3518357, 3.3470459
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0652790, 3.0717463
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9745064, 3.9884520
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1874847, 4.1805429
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6809295, upper bound: 1.6750281
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6594715, upper bound: 1.6964919
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8635263, 3.8727398
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6446495, 3.6486511
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4148531, 3.3951931
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3483701, 3.3505116
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0624084, 3.0746164
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9753513, 3.9876060
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1843824, 4.1836452
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6759057, upper bound: 1.6800201
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6544475, upper bound: 1.7014811
time: 4.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.7014813, upper bound: 1.6544474
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6800202, upper bound: 1.6759049
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6964921, upper bound: 1.6594716
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6750284, upper bound: 1.6809294
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.7022954, upper bound: 1.6536540
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6808337, upper bound: 1.6751119
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6973096, upper bound: 1.6586778
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6758453, upper bound: 1.6801345
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6801322, upper bound: 1.6758456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6586755, upper bound: 1.6973098
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6751120, upper bound: 1.6808335
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6536537, upper bound: 1.7022951
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6809295, upper bound: 1.6750281
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6594715, upper bound: 1.6964919
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6759057, upper bound: 1.6800201
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.27
Output dim: 9, lower bound: -1.6544475, upper bound: 1.7014811

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8644238, 3.8576355
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6399794, 3.6385040
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3999329, 3.4180856
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3435369, 3.3385234
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0663137, 3.0565257
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9798717, 3.9698734
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1866360, 4.1887703
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7014758, upper bound: 1.6538718
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6957903, upper bound: 1.6538789
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8668489, 3.8552103
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6425066, 3.6359768
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3984261, 3.4195910
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3406644, 3.3413963
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0687342, 3.0541053
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9821281, 3.9676170
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1880341, 4.1873732
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6800147, upper bound: 1.6753282
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6743290, upper bound: 1.6753359
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8629103, 3.8591485
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6410437, 3.6374397
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4027948, 3.4152231
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3400731, 3.3419886
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0634432, 3.0593958
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9807167, 3.9690275
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1835346, 4.1918736
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6957925, upper bound: 1.6538746
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6957874, upper bound: 1.6594673
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8653364, 3.8567233
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6435709, 3.6349125
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4012890, 3.4167285
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3371987, 3.3448615
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0658636, 3.0569754
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9829731, 3.9667711
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1849308, 4.1904755
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6743313, upper bound: 1.6753311
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6743261, upper bound: 1.6809277
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8548441, 3.8672152
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6342840, 3.6441989
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4065342, 3.4114833
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3354058, 3.3466554
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0602112, 3.0626278
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9769125, 3.9728327
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1941891, 4.1812172
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7022899, upper bound: 1.6530731
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6966098, upper bound: 1.6530777
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8572693, 3.8647900
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6368113, 3.6416721
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4050293, 3.4129891
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3325315, 3.3495283
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0626316, 3.0602074
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9791689, 3.9705763
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1955872, 4.1798201
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6808282, upper bound: 1.6745274
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6751487, upper bound: 1.6745318
time: 7.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8533316, 3.8687277
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6353483, 3.6431346
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4093971, 3.4086208
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3319402, 3.3501205
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0573406, 3.0654979
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9777584, 3.9719868
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1910868, 4.1843204
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6966120, upper bound: 1.6530755
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6966069, upper bound: 1.6586713
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8557558, 3.8663030
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6378756, 3.6406074
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4078913, 3.4101267
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3290677, 3.3529935
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0597610, 3.0630774
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9800148, 3.9697304
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1924839, 4.1829224
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6751510, upper bound: 1.6745297
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6751458, upper bound: 1.6801271
time: 7.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8663034, 3.8557563
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6406069, 3.6378760
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4101267, 3.4078913
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3529935, 3.3290672
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0630779, 3.0597610
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9697304, 3.9800148
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1829224, 4.1924839
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6801218, upper bound: 1.6757603
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6800505, upper bound: 1.6758356
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8687277, 3.8533316
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6431341, 3.6353488
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4086208, 3.4093971
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3501210, 3.3319402
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0654984, 3.0573406
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9719868, 3.9777579
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1843204, 4.1910868
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6586712, upper bound: 1.6966073
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6530755, upper bound: 1.6966117
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8647900, 3.8572693
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6416721, 3.6368113
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4129887, 3.4050288
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3495278, 3.3325324
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0602074, 3.0626311
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9705763, 3.9791689
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1798201, 4.1955872
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6745322, upper bound: 1.6751485
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6745271, upper bound: 1.6808279
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8672152, 3.8548441
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6441984, 3.6342840
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4114838, 3.4065342
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3466554, 3.3354053
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0626278, 3.0602107
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9728327, 3.9769125
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1812172, 4.1941891
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6530777, upper bound: 1.6966096
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6530726, upper bound: 1.7022896
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8567228, 3.8653359
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6349125, 3.6435709
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4167290, 3.4012890
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3448625, 3.3371992
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0569754, 3.0658631
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9667711, 3.9829736
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1904755, 4.1849308
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6809254, upper bound: 1.6743262
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6753316, upper bound: 1.6743313
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8591490, 3.8629107
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6374397, 3.6410437
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4152231, 3.4027948
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3419881, 3.3400722
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0593958, 3.0634432
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9690275, 3.9807167
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1918736, 4.1835337
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6594674, upper bound: 1.6957876
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6538746, upper bound: 1.6957924
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.8552103, 3.8668489
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.6359768, 3.6425061
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.4195919, 3.3984265
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0920582, 5.0920582
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.3413968, 3.3406644
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -3.0541058, 3.0687337
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.9676170, 3.9821281
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.1873732, 4.1880341
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.62 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.6592936515808105
rel_dist={9: [-1.702359651903449, 1.7023598330847598]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4557
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4557

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175743, upper bound: 1.5991171
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5991165, upper bound: 1.6175744
time: 10.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.66
Output dim: 9, lower bound: -1.6175743, upper bound: 1.5991171
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.66
Output dim: 9, lower bound: -1.5991165, upper bound: 1.6175744

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7441368, 3.7457471
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4999437, 3.5004821
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3319397, 3.3406768
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0543499, 5.0549192
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2555509, 3.2636557
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9546766, 2.9519024
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8458405, 3.8371477
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0817928, 4.0786095
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6168019, upper bound: 1.5991149
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175720, upper bound: 1.5983199
time: 5.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7457466, 3.7441363
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5004816, 3.4999437
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3406773, 3.3319387
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0549183, 5.0543499
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2636552, 3.2555504
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9519024, 2.9546762
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8371477, 3.8458405
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0786095, 4.0817928
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4597
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4597

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5983194, upper bound: 1.6175725
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5991142, upper bound: 1.6168021
time: 8.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 9, lower bound: -1.6168019, upper bound: 1.5991149
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 9, lower bound: -1.6175720, upper bound: 1.5983199
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 9, lower bound: -1.5983194, upper bound: 1.6175725
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 9, lower bound: -1.5991142, upper bound: 1.6168021

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7344103, 3.7278099
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5206690, 3.5163260
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3092947, 3.3236909
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0524654, 5.0535011
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2420974, 3.2432327
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9389400, 2.9309359
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8581514, 3.8469229
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0804434, 4.0837336
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6167999, upper bound: 1.5950164
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6127012, upper bound: 1.5991122
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7261992, 3.7360210
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5157881, 3.5212078
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3149538, 3.3180318
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0529308, 5.0530357
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2351279, 3.2502031
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9337091, 2.9361663
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8556156, 3.8494592
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0869169, 4.0772600
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175700, upper bound: 1.5942244
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6134720, upper bound: 1.5983175
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7360210, 3.7261992
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5212078, 3.5157881
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3180323, 3.3149533
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0530338, 5.0529318
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2502036, 3.2351274
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9361668, 2.9337091
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8494587, 3.8556151
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0772600, 4.0869169
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5983174, upper bound: 1.6134725
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5942240, upper bound: 1.6175704
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7278099, 3.7344103
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5163260, 3.5206690
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3236914, 3.3092942
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0534992, 5.0524664
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2432332, 3.2420979
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9309359, 2.9389400
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8469229, 3.8581514
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0837336, 4.0804434
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5991122, upper bound: 1.6127012
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5950160, upper bound: 1.6168003
time: 4.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.6167999, upper bound: 1.5950164
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.6127012, upper bound: 1.5991122
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.6175700, upper bound: 1.5942244
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.6134720, upper bound: 1.5983175
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.5983174, upper bound: 1.6134725
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.5942240, upper bound: 1.6175704
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.5991122, upper bound: 1.6127012
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.22
Output dim: 9, lower bound: -1.5950160, upper bound: 1.6168003

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7222891, 3.7308149
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5121412, 3.5184736
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3051195, 3.3106527
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0466232, 5.0446243
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2262206, 3.2383256
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9263172, 2.9263144
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8527193, 3.8472881
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0789270, 4.0666103
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175663, upper bound: 1.5758298
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5991753, upper bound: 1.5942206
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7308149, 3.7222896
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5184736, 3.5121412
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3106527, 3.3051200
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0446243, 5.0466242
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2383256, 3.2262201
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9263144, 2.9263172
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8472881, 3.8527193
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0666113, 4.0789270
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 15.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5875
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5875

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5942203, upper bound: 1.5991753
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5758295, upper bound: 1.6175668
time: 5.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.88 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 9, lower bound: -1.6175663, upper bound: 1.5758298
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.88
Output dim: 9, lower bound: -1.5991753, upper bound: 1.5942206
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.88
Output dim: 9, lower bound: -1.5942203, upper bound: 1.5991753
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 9, lower bound: -1.5758295, upper bound: 1.6175668

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7139730, 3.7245770
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5034685, 3.5119677
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3096437, 3.3138857
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0338058, 5.0350094
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2188363, 3.2284789
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9180145, 2.9200854
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8449841, 3.8414874
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0819187, 4.0707989
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 15.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175616, upper bound: 1.5749729
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6126180, upper bound: 1.5749783
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7245770, 3.7139735
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5119677, 3.5034690
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.3138857, 3.3096437
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0350094, 5.0338058
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2284780, 3.2188363
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9200859, 2.9180140
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8414879, 3.8449845
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0707998, 4.0819187
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7325597, 3.7325597
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5820
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5749784, upper bound: 1.6126184
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5749730, upper bound: 1.6175612
time: 5.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.69 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 9, lower bound: -1.6175616, upper bound: 1.5749729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 9, lower bound: -1.6126180, upper bound: 1.5749783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.69
Output dim: 9, lower bound: -1.5749784, upper bound: 1.6126184
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.69
Output dim: 9, lower bound: -1.5749730, upper bound: 1.6175612

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7068615, 3.7151017
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4962053, 3.5065174
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2709703, 3.2914882
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0341454, 5.0248728
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2015018, 3.2053695
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9160681, 2.9174919
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8347158, 3.8347836
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0461245, 4.0230770
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6998405, 3.7121735
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175520, upper bound: 1.5748932
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174821, upper bound: 1.5749626
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7151012, 3.7068610
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5065174, 3.4962053
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2914886, 3.2709699
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0248718, 5.0341463
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2053690, 3.2015014
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9174919, 2.9160681
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8347836, 3.8347158
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0230780, 4.0461245
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7121744, 3.6998405
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5749621, upper bound: 1.6174824
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5748928, upper bound: 1.6175541
time: 5.03 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.78 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.78
Output dim: 9, lower bound: -1.6175520, upper bound: 1.5748932
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.78
Output dim: 9, lower bound: -1.6174821, upper bound: 1.5749626
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.78
Output dim: 9, lower bound: -1.5749621, upper bound: 1.6174824
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.78
Output dim: 9, lower bound: -1.5748928, upper bound: 1.6175541

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7064590, 3.7201037
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5081711, 3.5055566
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2770023, 3.2910051
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0533237, 5.0233364
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2010975, 3.2103920
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9272728, 2.9165921
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8338051, 3.8461194
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0509367, 4.0226908
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6996231, 3.7148848
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175430, upper bound: 1.5715025
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5958140, upper bound: 1.5715199
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7068615, 3.7146997
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4952450, 3.5065174
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2704868, 3.2914882
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0326099, 5.0248728
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2015018, 3.2049665
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9151688, 2.9174919
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8347158, 3.8338737
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0457382, 4.0230770
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6998405, 3.7119570
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174732, upper bound: 1.5715718
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5957440, upper bound: 1.5715893
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7146997, 3.7118607
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5184765, 3.4952445
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2975216, 3.2704868
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0440426, 5.0326090
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2049656, 3.2065210
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9286938, 2.9151683
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8338737, 3.8460512
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0278873, 4.0457382
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7119570, 3.7025514
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5715889, upper bound: 1.5957445
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5715713, upper bound: 1.6174733
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7151012, 3.7064595
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5055571, 3.4962053
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2910061, 3.2709699
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0233364, 5.0341463
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.2053690, 3.2010984
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9165916, 2.9160681
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8347836, 3.8338056
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0226908, 4.0461245
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7121744, 3.6996236
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 961

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5715196, upper bound: 1.5958144
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5715020, upper bound: 1.6175434
time: 5.06 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.10 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.6175430, upper bound: 1.5715025
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.5958140, upper bound: 1.5715199
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.6174732, upper bound: 1.5715718
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.5957440, upper bound: 1.5715893
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.5715889, upper bound: 1.5957445
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.5715713, upper bound: 1.6174733
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.5715196, upper bound: 1.5958144
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.10
Output dim: 9, lower bound: -1.5715020, upper bound: 1.6175434

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6961985, 3.7162604
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5113163, 3.5055442
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2808447, 3.2909937
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0475521, 5.0211802
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1988840, 3.2044773
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9209185, 2.9142036
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8272572, 3.8436680
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0506535, 4.0219364
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6909103, 3.7116218
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175393, upper bound: 1.5709187
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6169602, upper bound: 1.5715004
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6966000, 3.7108564
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4983902, 3.5065041
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2743292, 3.2914758
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0268383, 5.0227175
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1992865, 3.1990519
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9088154, 2.9151034
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8281660, 3.8314223
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0454550, 4.0223217
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6911259, 3.7086940
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174697, upper bound: 1.5709887
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6168899, upper bound: 1.5715676
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7108564, 3.7016001
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5184641, 3.4983902
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2975092, 3.2743292
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0418873, 5.0268373
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1990519, 3.2043061
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9263058, 2.9088149
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8314219, 3.8395023
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0271330, 4.0454550
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7086945, 3.6938381
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5715672, upper bound: 1.6168902
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5709882, upper bound: 1.6174697
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7112579, 3.6961985
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5055437, 3.4993501
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2909937, 3.2748113
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0211811, 5.0283747
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1994553, 3.1988835
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9142036, 2.9097147
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8323317, 3.8272572
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0219374, 4.0458403
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7089100, 3.6909099
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4610
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 4610

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5714981, upper bound: 1.6169605
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5709188, upper bound: 1.6175405
time: 5.01 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 25.84 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.6175393, upper bound: 1.5709187
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.6169602, upper bound: 1.5715004
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.6174697, upper bound: 1.5709887
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.6168899, upper bound: 1.5715676
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.5715672, upper bound: 1.6168902
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.5709882, upper bound: 1.6174697
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.5714981, upper bound: 1.6169605
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 25.84
Output dim: 9, lower bound: -1.5709188, upper bound: 1.6175405

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6904755, 3.7036958
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5055418, 3.4928522
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2798147, 3.2887378
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0378017, 5.0167542
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1984520, 3.2035537
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9154372, 2.9022026
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8260937, 3.8411112
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0504379, 4.0214605
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6884575, 3.7062335
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6151058, upper bound: 1.5709168
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6175373, upper bound: 1.5684845
time: 5.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6836338, 3.7105350
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4986248, 3.4997616
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2785892, 3.2899642
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0431194, 5.0114298
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1979599, 3.2040434
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9089179, 2.9087229
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8247004, 3.8425026
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0501776, 4.0217199
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6855211, 3.7091675
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6145268, upper bound: 1.5714961
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6169581, upper bound: 1.5690648
time: 5.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6908770, 3.6982918
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4926157, 3.4938116
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2732992, 3.2892218
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0170879, 5.0182915
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1988554, 3.1981273
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9033322, 2.9031019
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8270025, 3.8288660
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0452394, 4.0218458
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6886740, 3.7033052
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6150371, upper bound: 1.5709865
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6174677, upper bound: 1.5685538
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6840363, 3.7051353
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4856977, 3.5007210
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2720737, 3.2904482
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0224018, 5.0129662
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1983633, 3.1986165
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8968139, 2.9096222
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8256092, 3.8302579
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0449791, 4.0221062
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6857376, 3.7062407
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.6144575, upper bound: 1.5715655
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6168878, upper bound: 1.5691337
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7051353, 3.6890354
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5126820, 3.4856982
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2964792, 3.2720737
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0321369, 5.0224037
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1986160, 3.2033820
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9208264, 2.8968139
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8302584, 3.8369455
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0269175, 4.0449791
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7062407, 3.6884494
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5691333, upper bound: 1.6168884
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5715652, upper bound: 1.6144578
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6982918, 3.6958733
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5057716, 3.4926157
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2952538, 3.2732997
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0374622, 5.0170870
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1981277, 3.2038760
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9143052, 2.9033327
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8288660, 3.8383369
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0266571, 4.0452394
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7033052, 3.6913843
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5685537, upper bound: 1.6174675
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5709862, upper bound: 1.6150368
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.7055368, 3.6836338
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4997616, 3.4866581
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2899637, 3.2725573
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0114269, 5.0239410
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1990194, 3.1979594
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9087234, 2.8977137
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8311672, 3.8247004
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0217209, 4.0453644
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7064571, 3.6855211
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5690647, upper bound: 1.6169584
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5714961, upper bound: 1.6145273
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6986933, 3.6904750
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.4928522, 3.4935751
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2887383, 3.2737837
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0167522, 5.0186243
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1985312, 3.1984525
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.9022031, 2.9042320
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8297749, 3.8260937
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0214605, 4.0456247
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.7035217, 3.6884580
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5798
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 4608

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5798

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5684847, upper bound: 1.6175376
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.5709168, upper bound: 1.6151055
time: 5.39 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 25.95 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6151058, upper bound: 1.5709168
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6175373, upper bound: 1.5684845
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6145268, upper bound: 1.5714961
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6169581, upper bound: 1.5690648
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6150371, upper bound: 1.5709865
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6174677, upper bound: 1.5685538
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6144575, upper bound: 1.5715655
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.6168878, upper bound: 1.5691337
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5691333, upper bound: 1.6168884
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5715652, upper bound: 1.6144578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5685537, upper bound: 1.6174675
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5709862, upper bound: 1.6150368
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5690647, upper bound: 1.6169584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5714961, upper bound: 1.6145273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5684847, upper bound: 1.6175376
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.95
Output dim: 9, lower bound: -1.5709168, upper bound: 1.6151055

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7212830, -5.1725097, -9.7212830, -5.1725097, -3.6814814, 3.6969509
1: -17.1234493, -13.3353291, -17.1234493, -13.3353291, -3.5020342, 3.4911108
2: -8.1695738, -4.2664623, -8.1695738, -4.2664623, -3.2782574, 3.2875676
3: -13.8282661, -8.7362080, -13.8282661, -8.7362080, -5.0249357, 4.9995966
4: -3.9074721, -0.2440012, -3.9074721, -0.2440012, -3.1513634, 3.1693287
5: -13.9941139, -9.9756889, -13.9941139, -9.9756889, -2.8886724, 2.8650637
6: -15.9595566, -11.4040337, -15.9595566, -11.4040337, -3.8145857, 3.8238969
7: -8.3881960, -4.1792936, -8.3881960, -4.1792936, -4.0387058, 4.0058165
8: -6.7207732, -2.9882135, -6.7207732, -2.9882135, -3.6965942, 3.7117157
9: 3.9066463, 6.5659399, 3.9066463, 6.5659399, -2.6592937, 2.6592937

Time for backsubstitution: 15.04 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.6592936515808105
rel_dist={9: [-1.6176245637311917, 1.6176249981850015]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2423.26 seconds
